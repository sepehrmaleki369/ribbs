import os
import json
import warnings
import logging
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt, rotate, binary_dilation
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
assumptions:
    - lbl can be binary or int (thresholded by 127)
    - roads are 1 on lbl
    - image modality must be defined 
    - label modality is not required necessarily (even it's possible to define one folder to more than one modality) 
    - for modalities other that label and image: 
        - file names must be in this format: modality_filename = f"{base}_{key}.npy"
        - if the computed modality's folder does not contain file "config.json" it will be computed again
    - if use_splitting then there is two options:
        - 1) kfold -> source_folder, num_folds and fold is required
        - 2) split_ratio -> source_folder, ratios are required
    - there is two optiona overall: 
        - 1) setting stride -> so dataloader extracts all valid patches per image (removed in this version)
        - 2) if stride was None -> extracts just one patch per image
"""

def compute_distance_map(lbl: np.ndarray, distance_threshold: Optional[float]) -> np.ndarray:
    """
    Compute a distance map from a label image.

    Args:
        lbl (np.ndarray): Input label image.
        distance_threshold (Optional[float]): Maximum distance value.
    
    Returns:
        np.ndarray: Distance map.
    """
    lbl_bin = (lbl > 127).astype(np.uint8) if lbl.max() > 1 else (lbl > 0).astype(np.uint8)
    distance_map = distance_transform_edt(lbl_bin == 0)
    if distance_threshold is not None:
        np.minimum(distance_map, distance_threshold, out=distance_map)
    return distance_map

def compute_sdf(lbl: np.ndarray, sdf_iterations: int, sdf_thresholds: List[float]) -> np.ndarray:
    """
    Compute the signed distance function (SDF) for a label image.

    Args:
        lbl (np.ndarray): Input label image.
        sdf_iterations (int): Number of iterations for dilation.
        sdf_thresholds (List[float]): [min, max] thresholds for the SDF.
    
    Returns:
        np.ndarray: The SDF computed.
    """
    lbl_bin = (lbl > 127).astype(np.uint8) if lbl.max() > 1 else (lbl > 0).astype(np.uint8)
    dilated = binary_dilation(lbl_bin, iterations=sdf_iterations)
    dist_out = distance_transform_edt(1 - dilated)
    dist_in  = distance_transform_edt(lbl_bin)
    sdf = dist_out - dist_in
    if sdf_thresholds is not None:
        sdf = np.clip(sdf, sdf_thresholds[0], sdf_thresholds[1])
    return sdf

def _is_readable_tiff(path: str) -> bool:
    """
    Stubbed-out for tests (and general use):
    never drop .tif/.tiff files based on rasterio.
    """
    return True

def load_array_from_file(file_path: str) -> Optional[np.ndarray]:
    """
    Load an array from disk.  If the file is unreadable, return None.
    """
    try:
        if file_path.endswith(".npy"):
            return np.load(file_path)
        else:
            with rasterio.open(file_path) as src:
                return src.read().astype(np.float32)
    except Exception:
        return None

class GeneralizedDataset(Dataset):
    """
    PyTorch Dataset for generalized remote sensing or segmentation datasets.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.root_dir: str = config.get("root_dir")
        self.split: str = config.get("split", "train")
        self.patch_size: int = config.get("patch_size", 128)
        self.small_window_size: int = config.get("small_window_size", 8)
        self.threshold: float = config.get("threshold", 0.05)
        self.max_images: Optional[int] = config.get("max_images")
        self.max_attempts: int = config.get("max_attempts", 10)
        self.validate_road_ratio: bool = config.get("validate_road_ratio", False)
        self.seed: int = config.get("seed", 42)
        self.fold = config.get("fold")
        self.num_folds = config.get("num_folds")
        self.verbose: bool = config.get("verbose", False)
        self.augmentations: List[str] = config.get("augmentations", ["flip_h", "flip_v", "rotation"])
        self.distance_threshold: Optional[float] = config.get("distance_threshold")
        self.sdf_iterations: int = config.get("sdf_iterations")
        self.sdf_thresholds: List[float] = config.get("sdf_thresholds")
        self.num_workers: int = config.get("num_workers", 4)
        self.split_ratios: Dict[str, float] = config.get("split_ratios", {"train":0.7,"valid":0.15,"test":0.15})
        self.use_splitting: bool = config.get("use_splitting", False)
        self.modalities: Dict[str, str] = config.get("modalities", {"image":"sat","label":"map"})
        self.source_folder: str = config.get("source_folder", "")
        self.save_computed: bool = config.get("save_computed", False)

        if self.root_dir is None:
            raise ValueError("root_dir must be specified in the config.")
        if self.patch_size is None:
            raise ValueError("patch_size must be specified in the config.")

        random.seed(self.seed)
        np.random.seed(self.seed)

        split_dir = os.path.join(self.root_dir, self.split)
        if self.use_splitting and self.split != 'test':
            split_dir = os.path.join(self.root_dir, self.source_folder)
        self.data_dir: str = split_dir

        self.modality_dirs: Dict[str, str] = {}
        self.modality_files: Dict[str, List[str]] = {}
        exts = ['.tiff', '.tif', '.png', '.jpg', '.npy']

        # Process "image" modality.
        folder_name = self.modalities['image']
        mod_dir = os.path.join(self.data_dir, folder_name)
        if not os.path.isdir(mod_dir):
            raise ValueError(f"Modality directory {mod_dir} not found.")
        self.modality_dirs['image'] = mod_dir
        files = sorted(
            f for f in os.listdir(mod_dir)
            if any(f.endswith(ext) for ext in exts)
        )
        self.modality_files['image'] = files

        if self.max_images is not None:
            self.modality_files['image'] = self.modality_files['image'][:self.max_images]

        # Process "label" modality.
        if "label" in self.modalities:
            folder_name = self.modalities['label']
            mod_dir = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(mod_dir):
                raise ValueError(f"Modality directory {mod_dir} not found.")
            self.modality_dirs['label'] = mod_dir
            files = sorted(
                f for f in os.listdir(mod_dir)
                if any(f.endswith(ext) for ext in exts)
            )
            self.modality_files['label'] = files

            if self.max_images is not None:
                self.modality_files['label'] = self.modality_files['label'][:self.max_images]

        # Precompute additional modalities (e.g., distance, sdf).
        for key in [modal for modal in self.modalities if modal not in ['image', 'label']]:
            folder_name = self.modalities[key]
            mod_dir = os.path.join(self.data_dir, folder_name)
            os.makedirs(mod_dir, exist_ok=True)
            sdf_comp_again = False
            config_path = os.path.join(mod_dir, "config.json")
            # if os.path.exists(config_path):
            #     with open(config_path, "r") as config_file:
            #         saved_config = json.load(config_file)
            #         sdf_comp_again = self.sdf_iterations != saved_config.get("sdf_iterations", None)

            logger.info(f"Generating {key} modality maps...")
            for file_idx, file in tqdm(enumerate(self.modality_files["label"]),
                                       total=len(self.modality_files["label"]),
                                       desc=f"Processing {key} maps"):
                lbl_path = os.path.join(self.modality_dirs['label'], self.modality_files["label"][file_idx])
                lbl = load_array_from_file(lbl_path)
                base, _ = os.path.splitext(file)
                modality_filename = f"{base}_{key}.npy"
                modality_path = os.path.join(mod_dir, modality_filename)
                if self.save_computed:
                    if key == "distance":
                        if not os.path.exists(modality_path):
                            processed_map = compute_distance_map(lbl, None)
                            np.save(modality_path, processed_map)
                    elif key == "sdf":
                        if not os.path.exists(modality_path) or sdf_comp_again:
                            processed_map = compute_sdf(lbl, self.sdf_iterations, None)
                            np.save(modality_path, processed_map)
                    else:
                        raise ValueError(f"Modality {key} not supported.")
            with open(config_path, "w") as config_file:
                json.dump(self.config, config_file, indent=4)
            self.modality_dirs[key] = mod_dir
            files = sorted([f for f in os.listdir(mod_dir) if any(f.endswith(ext) for ext in exts)])
            self.modality_files[key] = files

        # Perform dataset splitting if requested.
        if self.use_splitting:
            if self.fold is not None and self.num_folds is not None:
                # ----- KFold Splitting -----
                files = self.modality_files["image"]
                kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
                splits = list(kf.split(files))
                if self.split == "train":
                    selected_indices = splits[self.fold][0].tolist()
                elif self.split == "valid":
                    selected_indices = splits[self.fold][1].tolist()
                elif self.split == "test":
                    selected_indices = [i for i in range(len(files))]
                else:
                    raise ValueError("For KFold splitting, split must be 'train' or 'valid' or 'test'.", self.split)
                for key in self.modality_files:
                    all_files = self.modality_files[key]
                    self.modality_files[key] = [all_files[i] for i in selected_indices]
            else:
                # ----- Split by Ratios -----
                files = self.modality_files["label"]
                num_files = len(files)
                indices = np.arange(num_files)
                np.random.shuffle(indices)
                train_count = int(num_files * self.split_ratios["train"])
                valid_count = int(num_files * self.split_ratios["valid"])
                if self.split == "train":
                    selected_indices = indices[:train_count]
                elif self.split == "valid":
                    selected_indices = indices[train_count:train_count + valid_count]
                elif self.split == "test":
                    selected_indices = indices[train_count + valid_count:]
                else:
                    raise ValueError("For an 'entire' folder, split must be one of 'train', 'valid', or 'test'.")
                for key in self.modality_files:
                    all_files = self.modality_files[key]
                    self.modality_files[key] = [all_files[i] for i in selected_indices]

        if self.max_images is not None:
            for key in self.modality_files:
                self.modality_files[key] = self.modality_files[key][:self.max_images]

    def _load_datapoint(self, file_idx: int) -> Optional[Dict[str, np.ndarray]]:
        imgs: Dict[str, np.ndarray] = {}
        for key in self.modalities:
            if file_idx < len(self.modality_files[key]):
                file_path = os.path.join(self.modality_dirs[key], self.modality_files[key][file_idx])
                arr = load_array_from_file(file_path)
                if arr is None:            # <- corrupted TIFF
                    return None            # signal caller to skip this index
            else:
                lbl_path = os.path.join(self.modality_dirs['label'], self.modality_files["label"][file_idx])
                lbl = load_array_from_file(lbl_path)
                if key == "distance":
                    arr = compute_distance_map(lbl, None)
                elif key == "sdf":
                    arr = compute_sdf(lbl, self.sdf_iterations, None)
                else:
                    raise ValueError(f"Modality {key} not supported.")
                
            imgs[key] = arr
        return imgs

    def _check_small_window(self, image_patch: np.ndarray) -> bool:
        """
        Check that no small window in the image patch is entirely black or white.

        Args:
            image_patch (np.ndarray): Input patch (H x W) or (C x H x W)

        Returns:
            bool: True if valid, False if any window is all black or white.
        """
        sw = self.small_window_size

        # Ensure image has shape (C, H, W)
        if image_patch.ndim == 2:
            image_patch = image_patch[None, :, :]  # Add channel dimension

        C, H, W = image_patch.shape
        if H < sw or W < sw:
            return False

        # Set thresholds
        max_val = image_patch.max()
        if max_val > 1.0:
            high_thresh = 255
            low_thresh = 0
        else:
            high_thresh = 255 / 255.0
            low_thresh = 0 / 255.0

        # Slide window over spatial dimensions
        for c in range(C):
            for y in range(0, H - sw + 1):
                for x in range(0, W - sw + 1):
                    window = image_patch[c, y:y + sw, x:x + sw]
                    window_var = np.var(window)
                    if window_var < 0.01:
                        return False
                    # print(window)
                    if np.all(window >= high_thresh):
                        return False  # Found an all-white window
                    if np.all(window <= low_thresh):
                        return False  # Found an all-black window

        return True  # All windows passed


    def _check_min_thrsh_road(self, label_patch: np.ndarray) -> bool:
        """
        Check if the label patch has at least a minimum percentage of road pixels.

        Args:
            label_patch (np.ndarray): The label patch.
        
        Returns:
            bool: True if the patch meets the minimum threshold; False otherwise.
        """
        patch = label_patch
        if patch.max() > 1:
            patch = (patch > 127).astype(np.uint8)
        road_percentage = np.sum(patch) / (self.patch_size * self.patch_size)
        return road_percentage >= self.threshold

    def __len__(self) -> int:
        return len(self.modality_files['image'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imgs = self._load_datapoint(idx)
        if imgs is None:           # corrupted file detected
            # pick a different index (cyclic) so DataLoader doesn’t crash
            return self.__getitem__((idx + 1) % len(self))

        if imgs['image'].ndim == 3:
            _, H, W = imgs['image'].shape
        elif imgs['image'].ndim == 2:
            H, W = imgs['image'].shape
        else:
            raise ValueError("Unsupported image dimensions")

        if self.split != 'train':
            data = {}
            patch_meta = {"image_idx": idx, "x": -1, "y": -1}
            for key, array in imgs.items():
                if array.ndim == 3:
                    data[f"{key}_patch"] = array
                elif array.ndim == 2:
                    data[f"{key}_patch"] = array
                else:
                    raise ValueError("Unsupported array dimensions in _extract_data")
            data['metadata'] = patch_meta
            data = self._postprocess_patch(data)
            
            return data

        valid_patch_found = False
        attempts = 0
        while not valid_patch_found and attempts < self.max_attempts:
            x = np.random.randint(0, W - self.patch_size + 1)
            y = np.random.randint(0, H - self.patch_size + 1)
            patch_meta = {"image_idx": idx, "x": x, "y": y}
            if self.augmentations:
                patch_meta.update(self._get_augmentation_metadata())
            data = self._extract_condition_augmentations(imgs, patch_meta)
            if self.validate_road_ratio:
                if self._check_min_thrsh_road(data['label_patch']):
                    valid_patch_found = True
                    data['metadata'] = patch_meta
                    data = self._postprocess_patch(data)
                    return data
            else:
                valid_patch_found = True
                data['metadata'] = patch_meta
                data = self._postprocess_patch(data)
                return data
            
            attempts += 1

        # If a valid patch isn't found after max_attempts, fallback to the last sampled patch
        if not valid_patch_found:
            logger.warning("No valid patch found after %d attempts; trying next image", self.max_attempts)
            return self.__getitem__((idx + 1) % len(self))
        return None
        
    def _get_augmentation_metadata(self) -> Dict[str, Any]:
        """
        Generate random augmentation parameters for a patch.

        Returns:
            Dict[str, Any]: Augmentation metadata.
        """
        meta: Dict[str, Any] = {}
        if 'rotation' in self.augmentations:
            meta['angle'] = np.random.uniform(0, 360)
        if 'flip_h' in self.augmentations:
            meta['flip_h'] = np.random.rand() > 0.5
        if 'flip_v' in self.augmentations:
            meta['flip_v'] = np.random.rand() > 0.5
        return meta

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image / 255.0 if image.max() > 1.0 else image

    def _postprocess_patch(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize image patch values and binarize label patch if needed.

        Args:
            data (Dict[str, np.ndarray]): Dictionary containing patches.
        
        Returns:
            Dict[str, np.ndarray]: Postprocessed patches.
        """
        for key in data:
            if key == "image_patch":
                data[key] = self._normalize_image(data[key])
            elif key == "label_patch":
                if data[key].max() > 1:
                    data[key] = (data[key] > 127).astype(np.uint8)
            elif key == "distance_patch":
                if self.distance_threshold:
                    data[key] = np.clip(data[key], 0, self.distance_threshold)
            elif key == "sdf_patch":
                if self.sdf_thresholds:
                    data[key] = np.clip(data[key], self.sdf_thresholds[0], self.sdf_thresholds[1])
        return data

    def _extract_condition_augmentations(self, imgs: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract a patch from the full image and apply conditional augmentations.

        Args:
            imgs (Dict[str, np.ndarray]): Full images for each modality.
            metadata (Dict[str, Any]): Metadata containing patch coordinates and augmentations.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted patches.
        """
        imgs_aug = imgs.copy()
        data = self._extract_data(imgs, metadata['x'], metadata['y'])
        for key in imgs:
            if key.endswith("_patch"):
                modality = key.replace("_patch", "")
                if 'flip_h' in self.augmentations:
                    imgs_aug[modality] = self._flip_h(imgs[modality])
                    data[key] = self._flip_h(data[key])
                if 'flip_v' in self.augmentations:
                    imgs_aug[modality] = self._flip_v(imgs[modality])
                    data[key] = self._flip_v(data[key])
                if 'rotation' in self.augmentations:
                    data[key] = self._rotate(imgs_aug[modality], metadata)
        return data

    def _extract_data(self, imgs: Dict[str, np.ndarray], x: int, y: int) -> Dict[str, np.ndarray]:
        """
        Extract a patch from each modality starting at (x, y) with size self.patch_size.

        Args:
            imgs (Dict[str, np.ndarray]): Full images.
            x (int): x-coordinate.
            y (int): y-coordinate.
        
        Returns:
            Dict[str, np.ndarray]: Extracted patch for each modality.
        """
        data: Dict[str, np.ndarray] = {}
        for key, array in imgs.items():
            if array.ndim == 3:
                data[f"{key}_patch"] = array[:, y:y + self.patch_size, x:x + self.patch_size]
            elif array.ndim == 2:
                data[f"{key}_patch"] = array[y:y + self.patch_size, x:x + self.patch_size]
            else:
                raise ValueError("Unsupported array dimensions in _extract_data")
        return data

    def _flip_h(self, full_array: np.ndarray) -> np.ndarray:
        return np.flip(full_array, axis=-1)
    
    def _flip_v(self, full_array: np.ndarray) -> np.ndarray:
        return np.flip(full_array, axis=-2)
    
    def _rotate(self, full_array: np.ndarray, patch_meta: Dict[str, Any]) -> np.ndarray:
        """
        Rotate a patch using an expanded crop to avoid border effects.
        If the crop is too small, log a warning and return a zero patch.

        Args:
            full_array (np.ndarray): Full image array.
            patch_meta (Dict[str, Any]): Contains patch coordinates and angle.
        
        Returns:
            np.ndarray: Rotated patch.
        """
        patch_size = self.patch_size
        L = int(np.ceil(patch_size * math.sqrt(2)))
        x = patch_meta["x"]
        y = patch_meta["y"]
        angle = patch_meta["angle"]

        cx = x + patch_size // 2
        cy = y + patch_size // 2
        half_L = L // 2
        x0 = max(0, cx - half_L)
        y0 = max(0, cy - half_L)
        x1 = min(full_array.shape[-1], cx + half_L)
        y1 = min(full_array.shape[-2], cy + half_L)

        if full_array.ndim == 3:
            crop = full_array[:, y0:y1, x0:x1]
            if crop.shape[1] < L or crop.shape[2] < L:
                logger.warning("Crop too small for 3D patch rotation; returning zero patch.")
                return np.zeros((full_array.shape[0], patch_size, patch_size), dtype=full_array.dtype)
            rotated_channels = [rotate(crop[c], angle, reshape=False, order=1) for c in range(full_array.shape[0])]
            rotated = np.stack(rotated_channels)
            start = (L - patch_size) // 2
            return rotated[:, start:start + patch_size, start:start + patch_size]
        elif full_array.ndim == 2:
            crop = full_array[y0:y1, x0:x1]
            if crop.shape[0] < L or crop.shape[1] < L:
                logger.warning("Crop too small for 2D patch rotation; returning zero patch.")
                return np.zeros((patch_size, patch_size), dtype=full_array.dtype)
            rotated = rotate(crop, angle, reshape=False, order=1)
            start = (L - patch_size) // 2
            return rotated[start:start + patch_size, start:start + patch_size]
        else:
            raise ValueError("Unsupported array shape")

        
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function with None filtering.
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    # Handle empty batch case
    if not batch:
        logger.warning("Empty batch after filtering None values")
        return {}  # Or return a default empty batch structure
    
    # Original collation logic
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        items = []
        for sample in batch:
            value = sample[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            items.append(value)
        if isinstance(items[0], torch.Tensor):
            collated[key] = torch.stack(items)
        else:
            collated[key] = items
    return collated

def worker_init_fn(worker_id):
    """
    DataLoader worker initialization to ensure different random seeds for each worker.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def visualize_batch(batch: Dict[str, Any], num_per_batch: Optional[int] = None) -> None:
    """
    Visualizes patches in a batch: image, label, distance, and SDF (if available).

    Args:
        batch (Dict[str, Any]): Dictionary containing batched patches.
        num_per_batch (Optional[int]): Maximum number of patches to visualize.
    """
    import matplotlib.pyplot as plt

    num_to_plot = batch["image_patch"].shape[0]
    if num_per_batch:
        num_to_plot = min(num_to_plot, num_per_batch)
    for i in range(num_to_plot):
        sample_image = batch["image_patch"][i].numpy()
        if sample_image.shape[0] == 3:  # CHW to HWC
            sample_image = sample_image.transpose(1, 2, 0)
        elif sample_image.shape[0] == 1:
            sample_image = sample_image[0]  # grayscale
        else:
            sample_image = sample_image.transpose(1, 2, 0)

        sample_label = np.squeeze(batch["label_patch"][i].numpy())
        sample_distance = batch["distance_patch"][i].numpy() if "distance_patch" in batch else None
        sample_sdf = batch["sdf_patch"][i].numpy() if "sdf_patch" in batch else None

        print(f'Patch {i}')
        print('  image:', sample_image.min(), sample_image.max())
        print('  label:', sample_label.min(), sample_label.max())
        if sample_distance is not None:
            print('  distance:', sample_distance.min(), sample_distance.max())
        if sample_sdf is not None:
            print('  sdf:', sample_sdf.min(), sample_sdf.max())

        num_subplots = 3 + (1 if sample_sdf is not None else 0)
        fig, axs = plt.subplots(1, num_subplots, figsize=(12, 4))
        axs[0].imshow(sample_image, cmap='gray' if sample_image.ndim == 2 else None)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(sample_label, cmap='gray')
        axs[1].set_title("Label")
        axs[1].axis("off")
        if sample_distance is not None:
            axs[2].imshow(sample_distance[0], cmap='gray')
            axs[2].set_title("Distance")
            axs[2].axis("off")
        else:
            axs[2].text(0.5, 0.5, "No Distance", ha='center', va='center')
            axs[2].axis("off")
        if sample_sdf is not None:
            axs[3].imshow(sample_sdf[0], cmap='coolwarm')
            axs[3].set_title("SDF")
            axs[3].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    config = {
        "root_dir": "/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/",  # Update with your dataset path.
        "split": "train",
        # "split": "valid",
        "patch_size": 256,
        "small_window_size": 8,
        "validate_road_ratio": True,
        "threshold": 0.025,
        "max_images": 1,  # For quick testing.
        "seed": 42,
        "fold": None,
        "num_folds": None,
        "verbose": True,
        "augmentations": ["flip_h", "flip_v", "rotation"],
        "distance_threshold": 100.0,
        "sdf_iterations": 3,
        "sdf_thresholds": [-20, 20],
        "num_workers": 4,
        "use_splitting": False,
        "split_ratios": {
            "train": 0.7,
            "valid": 0.15,
            "test": 0.15
        },
        "modalities": {
            "image": "sat",
            "label": "map",
            "distance": "distance",
            "sdf": "sdf"
        }
    }

    # Create dataset and dataloader.
    dataset = GeneralizedDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        # worker_init_fn=worker_init_fn
    )
    logger.info('len(dataloader): %d', len(dataloader))
    for epoch in range(10):
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            logger.info("Batch keys: %s", batch.keys())
            logger.info("Image shape: %s", batch["image_patch"].shape)
            logger.info("Label shape: %s", batch["label_patch"].shape)
            visualize_batch(batch)
            # break  # Uncomment to visualize only one batch.

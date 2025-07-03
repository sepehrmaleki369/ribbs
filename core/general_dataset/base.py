from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import os
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from core.general_dataset.io          import load_array_from_file
from core.general_dataset.modalities  import compute_distance_map, compute_sdf
from core.general_dataset.patch_validity       import check_min_thrsh_road
from core.general_dataset.augments    import get_augmentation_metadata, extract_condition_augmentations
from core.general_dataset.collate     import custom_collate_fn, worker_init_fn
from core.general_dataset.normalizations import normalize_image
from core.general_dataset.visualizations import visualize_batch
from core.general_dataset.logger import logger


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
                data[key] = normalize_image(data[key])
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
                patch_meta.update(get_augmentation_metadata(self.augmentations))
            data = extract_condition_augmentations(imgs, patch_meta, self.patch_size, self.augmentations)
            if self.validate_road_ratio:
                if check_min_thrsh_road(data['label_patch'], self.patch_size, self.threshold):
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
        "distance_threshold": 15.0,
        "sdf_iterations": 3,
        "sdf_thresholds": [-7, 7],
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

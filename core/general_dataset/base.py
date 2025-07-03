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
from core.general_dataset.visualizations import visualize_batch_2d, visualize_batch_3d
from core.general_dataset.splits import Split
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
    - Read Split explanation for more details
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
        self.source_folder: str = config.get("source_folder", "")
        self.save_computed: bool = config.get("save_computed", False)
        self.base_modalities = config.get('base_modalities')
        self.compute_again_modalities = config.get('compute_again_modalities', False)
        self.data_dim    = config.get("data_dim", 2)
        self.split_cfg = config["split_cfg"]
        self.split_cfg['seed'] = self.seed
        self.split_cfg['base_modalities'] = self.base_modalities
        splitter = Split(self.split_cfg)
        self.patch_size_z = config.get("patch_size_z", 1)
        

        if self.root_dir is None:
            raise ValueError("root_dir must be specified in the config.")
        if self.patch_size is None:
            raise ValueError("patch_size must be specified in the config.")
        if self.data_dim not in (2, 3):
            raise ValueError(f"data_dim must be 2 or 3, got {self.data_dim}")
        if self.data_dim == 3 and self.patch_size_z < 2:
            raise ValueError("patch_size_z must > 1 for 3D")

        
        random.seed(self.seed)
        np.random.seed(self.seed)

        split_cfg = config["split_cfg"]
        split_cfg['seed'] = self.seed
        splitter = Split(split_cfg)
        self.modality_files: Dict[str, List[str]] = splitter.get_split(self.split)
        self.modalities = list(self.modality_files.keys())
        
        # sanity check
        if 'image' not in self.modality_files or 'label' not in self.modality_files:
            raise ValueError("Split must define both 'image' and 'label' modalities in split_cfg.")
        if 'image' in self.modality_files and 'label' in self.modality_files:
            assert len(self.modality_files['image']) == len(self.modality_files['label']), f"len(images): {len(self.modality_files['image'])}, len(labels): {len(self.modality_files['label'])}"

        # exts = ['.tiff', '.tif', '.png', '.jpg', '.npy']

        if self.max_images is not None:
            for key in self.modality_files:
                self.modality_files[key] = self.modality_files[key][:self.max_images]

        # Precompute additional modalities (e.g., distance, sdf).
        if self.save_computed:
            for key in [modal for modal in self.modalities if modal not in ['image', 'label']]:
                logger.info(f"Generating {key} modality maps...")
                for file_idx, file in tqdm(enumerate(self.modality_files["label"]),
                                        total=len(self.modality_files["label"]),
                                        desc=f"Processing {key} maps"):
                    lbl = load_array_from_file(self.modality_files['label'][file_idx])
                    modality_path = self.modality_files[key][file_idx]
                    dirpath = os.path.dirname(modality_path)
                    os.makedirs(dirpath, exist_ok=True)
                    if key == "distance":
                        if not os.path.exists(modality_path) or self.compute_again_modalities:
                            processed_map = compute_distance_map(lbl, None)
                            np.save(modality_path, processed_map)
                    elif key == "sdf":
                        if not os.path.exists(modality_path) or self.compute_again_modalities:
                            processed_map = compute_sdf(lbl, self.sdf_iterations, None)
                            np.save(modality_path, processed_map)
                    else:
                        raise ValueError(f"Modality {key} not supported.")

    def _load_datapoint(self, file_idx: int) -> Optional[Dict[str, np.ndarray]]:
        imgs: Dict[str, np.ndarray] = {}
        for key in self.modalities:
            if os.path.exists(self.modality_files[key][file_idx]):
                arr = load_array_from_file(self.modality_files[key][file_idx])
                if arr is None:            # <- corrupted TIFF
                    return None            # signal caller to skip this index
            else:
                lbl = load_array_from_file(self.modality_files["label"][file_idx])
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
        
        img = imgs['image']
        if self.data_dim == 3:
            # expect shape (C,D,H,W) or (D,H,W)
            if img.ndim == 4:
                _, D, H, W = img.shape
            elif img.ndim == 3:
                D, H, W = img.shape
            else:
                raise ValueError(f"Expected 3- or 4-dim image for data_dim=3; got {img.ndim}D")
        else:  # 2-D
            if img.ndim == 2:
                D, H, W = 1, *img.shape
            elif img.ndim == 3:
                _, H, W = img.shape
                D = 1
            else:
                raise ValueError(f"Expected 2- or 3-dim image for data_dim=2; got {img.ndim}D")

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
            if self.data_dim == 3:
                z = np.random.randint(0, D - self.patch_size_z + 1)
            else:
                z = 0
            # patch_meta = {"image_idx": idx, "x": x, "y": y}
            patch_meta = {"image_idx": idx, "x": x, "y": y, "z": z}

            if self.augmentations:
                patch_meta.update(
                    get_augmentation_metadata(self.augmentations, self.data_dim)
                )
            data = extract_condition_augmentations(
                imgs, patch_meta,
                patch_size_xy=self.patch_size,
                patch_size_z=(self.patch_size_z if self.data_dim == 3 else 1),
                augmentations=self.augmentations,
                data_dim=self.data_dim
            )


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
    # split_cfg = {
    #     "mode": "folder",
    #     "source_folders": [
    #     {
    #         "path": "/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset",
    #         "layout": "folders",
    #         "modalities": {
    #             "image": {"folder": "sat"},
    #             "label": {"folder": "label"},
    #             "sdf":   {"folder": "sdf"}  
    #         },
    #         "splits": {
    #             "train": "train",
    #             "valid": "valid",
    #             "test":  "test"
    #         }
    #     },
    # ]
    # }
    # config = {
    #     "root_dir": "/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/",  # Update with your dataset path.
    #     "split": "train",
    #     # "split": "valid",
    #     "split_cfg": split_cfg,
    #     "patch_size": 256,
    #     "small_window_size": 8,
    #     "validate_road_ratio": True,
    #     "threshold": 0.025,
    #     "max_images": 5,  # For quick testing.
    #     "seed": 42,
    #     "fold": None,
    #     "num_folds": None,
    #     "verbose": True,
    #     "augmentations": ["flip_h", "flip_v", 
    #                       "rotation"],
    #     "distance_threshold": 15.0,
    #     "sdf_iterations": 3,
    #     "sdf_thresholds": [-7, 7],
    #     "num_workers": 4,
    #     "split_ratios": {
    #         "train": 0.7,
    #         "valid": 0.15,
    #         "test": 0.15
    #     },
    #     "modalities": {
    #         "image": "sat",
    #         "label": "map",
    #         "distance": "distance",
    #         "sdf": "sdf"
    #     },
    #     "base_modalities": ['image', 'label'],
    #     'compute_again_modalities': False,
    #     'save_computed': True,
    #     "data_dim": 2,
    #     "patch_size_z": 16,
    # }

    split_cfg = {
        "seed": 42,
        "sources": [
            {
                "type": "ratio",
                "path": "/home/ri/Desktop/Projects/Datasets/AL175",
                "layout": "flat",
                "modalities": {
                    "image":    {"pattern": r"^cube_.*\.npy$"},
                    "label":    {"pattern": r"^lbl_.*\.npy$"},
                    "distance": {"pattern": r"^distlbl_.*\.npy$"},
                },
                "ratios": {
                    "train": 0.7,
                    "valid": 0.15,
                    "test":  0.15,
                }
            }
        ]
    }

    config = {
        "root_dir": "/home/ri/Desktop/Projects/Datasets/AL175",
        "split": "train",                # one of "train","valid","test"
        "split_cfg": split_cfg,

        "data_dim": 3,                   # your .npy volumes are 3D
        "patch_size": 64,                # XY window size
        "patch_size_z": 16,              # Z-depth

        "augmentations": ["flip_h","flip_v","flip_d","rotation"],
        "validate_road_ratio": False,    # set True if you want to enforce label coverage
        "threshold": 0.05,

        "distance_threshold": None,      # clip distance map if desired
        "sdf_iterations": 3,             # only if you compute SDF
        "sdf_thresholds": [-5, 5],       # ditto

        "save_computed": False,          # we already have distlbl_*.npy
        "compute_again_modalities": False,

        "max_images": None,
        "max_attempts": 10,
        "seed": 42,
        "num_workers": 4,
        "verbose": True,
        "base_modalities": ['image', 'label']
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
            if config["data_dim"] == 2:
                visualize_batch_2d(batch, num_per_batch=2)
            else:
                visualize_batch_3d(batch, projection='max', num_per_batch=2)

            # break  # Uncomment to visualize only one batch.

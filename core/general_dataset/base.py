from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
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
from core.general_dataset.collate     import custom_collate_fn, worker_init_fn
from core.general_dataset.normalizations import normalize_image
from core.general_dataset.augmentations import augment_image
from core.general_dataset.visualizations import visualize_batch_2d, visualize_batch_3d
from core.general_dataset.splits import Split
from core.general_dataset.logger import logger
import torch

def _to_tensor(obj):
    """Convert numpy ↦ torch (shared memory) but keep others unchanged."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)          # 0-copy, preserves shape/dtype
    return obj

class GeneralizedDataset(Dataset):
    """
    PyTorch Dataset for generalized remote sensing or segmentation datasets.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
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
        self.order_ops: List[str] = config.get("order_ops", ["crop", "aug", "norm"])
        self.norm_cfg: Dict[str, Optional[Dict[str, Any]]] = config.get("normalization", {})
        self.aug_cfg: Dict[str, Optional[Dict[str, Any]]] = config.get("augmentation", None)
        assert set(self.order_ops) == {"crop", "aug", "norm"}, \
                       f"order_ops must be a permutation of ['crop','aug','norm'], got {self.order_ops}"

        if self.patch_size is None:
            raise ValueError("patch_size must be specified in the config.")
        if self.data_dim not in (2, 3):
            raise ValueError(f"data_dim must be 2 or 3, got {self.data_dim}")
        if self.data_dim == 3 and config.get("patch_size_z", 1) < 2:
            raise ValueError("patch_size_z must > 1 for 3D")

        random.seed(self.seed)
        np.random.seed(self.seed)

        splitter = Split(self.split_cfg, self.base_modalities)
        self.modality_files: Dict[str, List[str]] = splitter.get_split(self.split)
        self.modalities = list(self.modality_files.keys())
        if 'image' not in self.modality_files or 'label' not in self.modality_files:
            raise ValueError("Split must define both 'image' and 'label' modalities in split_cfg.")
        assert len(self.modality_files['image']) == len(self.modality_files['label']), (
            f"len(images): {len(self.modality_files['image'])}, len(labels): {len(self.modality_files['label'])}")

        if self.max_images is not None:
            for key in self.modality_files:
                self.modality_files[key] = self.modality_files[key][:self.max_images]
                # print(key, 'Max Data Point:', len(self.modality_files[key]))
        # Precompute additional modalities if requested
        if self.save_computed:
            for key in [m for m in self.modalities if m not in ['image', 'label']]:
                logger.info(f"Generating {key} modality maps...")
                for file_idx, _ in tqdm(list(enumerate(self.modality_files['label'])),
                                        total=len(self.modality_files['label']),
                                        desc=f"Processing {key} maps"):
                    lbl = load_array_from_file(self.modality_files['label'][file_idx])
                    modality_path = self.modality_files[key][file_idx]
                    os.makedirs(os.path.dirname(modality_path), exist_ok=True)
                    if key == 'distance':
                        if not os.path.exists(modality_path) or self.compute_again_modalities:
                            processed = compute_distance_map(lbl, None)
                            np.save(modality_path, processed)
                    elif key == 'sdf':
                        if not os.path.exists(modality_path) or self.compute_again_modalities:
                            processed = compute_sdf(lbl, self.sdf_iterations, None)
                            np.save(modality_path, processed)
                    else:
                        raise ValueError(f"Unsupported modality {key}")

    
    def _load_datapoint(self, file_idx: int) -> Optional[Dict[str, np.ndarray]]:
        imgs: Dict[str, np.ndarray] = {}
        for key in self.modalities:
            path = self.modality_files[key][file_idx]
            if os.path.exists(path):
                arr = load_array_from_file(path)
                if arr is None:
                    return None
            else:
                lbl = load_array_from_file(self.modality_files['label'][file_idx])
                if key == 'distance':
                    arr = compute_distance_map(lbl, None)
                elif key == 'sdf':
                    arr = compute_sdf(lbl, self.sdf_iterations, None)
                else:
                    raise ValueError(f"Unsupported modality {key}")
            imgs[key] = arr
        if self.verbose:
            print('imgs[key] shape:', imgs[key].shape)
        return imgs

    def normalize_data(self, data):
        normalized_image = {}
        for key, arr in list(data.items()):
            cfg = self.norm_cfg.get(key, None)
            if cfg:
                method = cfg.get('method', None)
                params = {k: v for k, v in cfg.items() if k != 'method'}
                normalized_image[key] = normalize_image(arr, method=method, **params)
            else:
                normalized_image[key] = arr.copy()
        return normalized_image
    
    def augment_data(self, normalized_image):
        augmented_image = {}
        for key, arr in list(normalized_image.items()):
            # print('aug', key)
            if self.aug_cfg:
                # print('aug', self.aug_cfg)
                rng = np.random.RandomState(self.seed)
                augmented_image[key] = augment_image(arr, self.aug_cfg, self.data_dim, rng)
            else:
                augmented_image[key] = arr
        return augmented_image

    @staticmethod
    def _pad_reflect(arr: np.ndarray,
                     pad_before: Tuple[int, ...],
                     pad_after:  Tuple[int, ...]) -> np.ndarray:
        pads = tuple(zip(pad_before, pad_after))
        return np.pad(arr, pads, mode="reflect")

    def crop(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.split != "train":
            return data  # keep full image / volume for inference

        # ---------- determine crop coordinates from the first key ----------
        sample_key = next(k for k in data)
        sample = data[sample_key]

        if self.data_dim == 2:
            H, W = sample.shape
            ph = max(0, self.patch_size - H)
            pw = max(0, self.patch_size - W)

            # pad if necessary
            if ph > 0 or pw > 0:
                for k in data:
                    arr = data[k]
                    data[k] = self._pad_reflect(
                        arr,
                        pad_before=(ph // 2, pw // 2),
                        pad_after=(ph - ph // 2, pw - pw // 2),
                    )
                H += ph
                W += pw

            # random top-left corner
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)

            # crop every modality
            for k in data:
                data[k] = data[k][
                    top : top + self.patch_size,
                    left: left + self.patch_size,
                ]

        else:  # ----------------------------- 3-D ---------------------------
            patch_z = self.config.get("patch_size_z", 1)
            D, H, W = sample.shape

            pd = max(0, patch_z       - D)
            ph = max(0, self.patch_size - H)
            pw = max(0, self.patch_size - W)

            if pd or ph or pw:
                for k in data:
                    arr = data[k]
                    data[k] = self._pad_reflect(
                        arr,
                        pad_before=(pd // 2, ph // 2, pw // 2),
                        pad_after=(pd - pd // 2, ph - ph // 2, pw - pw // 2),
                    )
                D += pd
                H += ph
                W += pw

            front = random.randint(0, D - patch_z)
            top   = random.randint(0, H - self.patch_size)
            left  = random.randint(0, W - self.patch_size)

            for k in data:
                data[k] = data[k][
                    front : front + patch_z,
                    top   : top   + self.patch_size,
                    left  : left  + self.patch_size,
                ]

        return data

    def _postprocess_patch(self, data: Dict[str, np.ndarray], augment: bool) -> Dict[str, np.ndarray]:
        # ------------------------------------------------------------
        # 1. Run the ops in user-defined order
        # ------------------------------------------------------------
        op = {
            "crop": lambda d: self.crop(d),
            "aug":  lambda d: self.augment_data(d) if augment else d,
            "norm": lambda d: self.normalize_data(d),
        }
        for step in self.order_ops:
            data = op[step](data)
            # if step == 'norm':
                # print('after norm', data['label'].min(), data['label'].max())

        # ------------------------------------------------------------
        # 2. Mark everything as *_patch  (↓ this is the only new line)
        # ------------------------------------------------------------
        data = {f"{k}_patch": v for k, v in data.items()}

        # ------------------------------------------------------------
        # 3. Add channel dim for PyTorch
        # ------------------------------------------------------------
        for k, arr in list(data.items()):
            if arr.ndim == 2:
                data[k] = arr[None, ...]             # (1, H, W)
            elif arr.ndim == 3 and self.data_dim == 3:
                data[k] = arr[None, ...]             # (1, D, H, W)
        return data


    def __len__(self) -> int:
        return len(self.modality_files['image'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imgs = self._load_datapoint(idx)
        if imgs is None:
            return self.__getitem__((idx + 1) % len(self))

        # Validation/test: returnimage_idx full image as one patch
        if self.split != 'train':
            return self._postprocess_patch(imgs, augment=False)

        # Training: random crop
        attempts = 0
        while attempts < self.max_attempts:
            data = self._postprocess_patch(imgs, augment=True)
            for k, v in data.items():
                data[k] = _to_tensor(v)

            # Validate road ratio if needed
            if not self.validate_road_ratio or check_min_thrsh_road(data['label_patch'], self.patch_size, self.threshold):
                return data
            attempts += 1

        logger.warning("No valid patch found after %d attempts on image %d; skipping.", self.max_attempts, idx)
        return self.__getitem__((idx + 1) % len(self))

if __name__ == "__main__":
    split_cfg = {
        "seed": 42,
        "sources": [
            {
                "type": "folder",
                "path": "/home/ri/Desktop/Projects/Datasets/RRRR/dataset",
                "layout": "folders",
                "modalities": {
                    "image":    {"folder": "sat"},
                    "label":    {"folder": "label"},
                    "distance": {"folder": "distance"},
                    "sdf":      {"folder": "sdf"},
                },
                "splits": {
                    "train": "train",
                    "valid": "valid",
                    "test":  "test",
                }
            }
        ]
    }

    # ------------------------------------
    # full config for GeneralizedDataset
    # ------------------------------------
    config = {
        # which split to load
        "split": "train",

        # pass the splitter cfg here
        "split_cfg": split_cfg,

        # patch extraction params
        "patch_size": 256,
        "patch_size_z": 1,             # keep at 1 for 2D data

        # optional small‐window check
        "small_window_size": 8,

        # require a minimum fraction of road pixels in each patch
        "validate_road_ratio": True,
        "threshold": 0.025,

        # if you want to limit to just N images (for debugging), set here
        "max_images": 5,

        # random seeds & workers
        "seed": 42,
        "num_workers": 4,
        "verbose": True,

        # train‐time augmentations
        "augmentations": ["flip_h", "flip_v", "rotation"],

        # distance & SDF modalities: compute if missing
        "distance_threshold": 15.0,
        "sdf_iterations": 3,
        "sdf_thresholds": [-7, 7],
        "compute_again_modalities": False,
        "save_computed": True,

        # only used if you ever switch to ratio‐based splitting
        "split_ratios": {"train": 0.7, "valid": 0.15, "test": 0.15},

        # these are the “base” modalities that must all be present
        "base_modalities": ["image", "label"],

        # 2D vs 3D
        "data_dim": 2,

        "normalization": {
            "image":    {"method":  "minmax",
                    "old_min": 0,     # your supplied min
                    "old_max": 255.0,   # your supplied max
                    "new_min": 0.0,
                    "new_max": 1.0},

            # "distance": {"method": "zscore"},
            "sdf":      {},            # empty → no normalization
            "label":    None,          # or null in JSON → skip
        },
        "augmentation_params": {
            "scale":             {"min": 1.5,  "max": 5.8},
            # "elastic":           {"alpha_min": 5.0,  "alpha_max": 10.0,
                                # "sigma_min": 3.0,  "sigma_max": 6.0},
            "brightness_contrast":{"alpha_min": 0.9,"alpha_max": 1.1,
                                "beta_min": -30.0,"beta_max": 30.0},
            "gamma":             {"min": 0.7,  "max": 1.5},
            # "gaussian_noise":    {"min": 0.01, "max": 0.03},
            # "gaussian_blur":     {"min": 0.5,  "max": 1.5},
            # "bias_field":        {"min": 0.2,  "max": 0.4},
            # "rotation":          {"min": 0.0,  "max": 360.0},
            # flips and flip_d remain Bernoulli(0.5), no extra params needed
        },
    }

    # split_cfg = {
    #     "seed": 42,
    #     "sources": [
    #         {
    #             "type": "ratio",
    #             "path": "/home/ri/Desktop/Projects/Datasets/AL175",
    #             "layout": "flat",
    #             "modalities": {
    #                 "image":    {"pattern": r"^cube_(.*)\.npy$"},
    #                 "label":    {"pattern": r"^lbl_(.*)\.npy$"},
    #                 "distance": {"pattern": r"^distlbl_(.*)\.npy$"},
    #             },
    #             "ratios": {
    #                 "train": 0.7,
    #                 "valid": 0.15,
    #                 "test":  0.15,
    #             }
    #         },
    #         {
    #             "path": "/home/ri/Desktop/Projects/Datasets/AL175",
    #             "layout": "flat",
    #             "modalities": {
    #                 "image":    {"pattern": r"^cube_(.*)\.npy$"},
    #                 "label":    {"pattern": r"^lbl_(.*)\.npy$"},
    #                 "distance": {"pattern": r"^distlbl_(.*)\.npy$"},
    #             },

    #             "type": "kfold",
    #             "num_folds": 5,
    #             "fold_idx": 0,
    #             "test_source":{
    #                 "type": "ratio",
    #                 "path": "/home/ri/Desktop/Projects/Datasets/AL175",
    #                 "layout": "flat",
    #                 "modalities": {
    #                     "image":    {"pattern": r"^cube_(.*)\.npy$"},
    #                     "label":    {"pattern": r"^lbl_(.*)\.npy$"},
    #                     "distance": {"pattern": r"^distlbl_(.*)\.npy$"},
    #                 },
    #                 "ratios": {
    #                     "train": 0.01,
    #                     "valid": 0.9,
    #                     "test":  0.01,
    #                 }
    #             }
    #         }
    #     ]
    # }

    # config = {
    #     "split": "train",                # one of "train","valid","test"
    #     "split_cfg": split_cfg,

    #     "data_dim": 3,                   # your .npy volumes are 3D
    #     "patch_size": 90,                # XY window size
    #     "patch_size_z": 90,              # Z-depth

    #     "augmentations": ["flip_h","flip_v","flip_d","rotation"],
    #     "validate_road_ratio": False,    # set True if you want to enforce label coverage
    #     "threshold": 0.05,

    #     "distance_threshold": None,      # clip distance map if desired
    #     "sdf_iterations": 3,             # only if you compute SDF
    #     "sdf_thresholds": [-5, 5],       # ditto

    #     "save_computed": True,          # we already have distlbl_*.npy
    #     "compute_again_modalities": False,

    #     "max_images": None,
    #     "max_attempts": 10,
    #     "seed": 42,
    #     "num_workers": 4,
    #     "verbose": True,
    #     "base_modalities": ['image', 'label']
    # }

    import yaml
    # with open('/home/ri/Desktop/Projects/Codebase/configs/dataset/main.yaml', 'w') as f_out:
        # yaml.dump(config, f_out)
    with open('./configs/dataset/AL175_15.yaml', 'r') as f:
    # with open('/home/ri/Desktop/Projects/Codebase/configs/dataset/massroads.yaml', 'r') as f:
        config = yaml.safe_load(f)

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

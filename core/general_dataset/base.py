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
from core.general_dataset.augmentations import augment_images
from core.general_dataset.crop import bigger_crop, center_crop
from core.general_dataset.visualizations import visualize_batch_2d, visualize_batch_3d
from core.general_dataset.splits import Split
from core.general_dataset.logger import logger
from core.general_dataset.io import to_tensor as _to_tensor
import torch

# --------------- helpers ----------------
def _merge_default(op: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    """Return op with default values filled in (shallow merge)."""
    merged = {**default, **op}            # op wins on conflict
    # modalities / interpolation need nested merge:
    for key in ("modalities", "interpolation"):
        if key not in merged and key in default:
            merged[key] = default[key]
    return merged


def _expand_aug_list(raw_list: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Consume the first element if it’s a `defaults:` block, clone merged ops."""
    if not raw_list or "defaults" not in raw_list[0]:
        return raw_list                    # nothing special
    defaults = raw_list[0]["defaults"]
    return [_merge_default(op, defaults) for op in raw_list[1:]]


class GeneralizedDataset(Dataset):
    """
    PyTorch Dataset for generalized remote sensing or segmentation datasets.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._epoch = 0
        self.config = config
        self.split: str = config.get("split", "train")
        self.patch_size: List[int] = config.get("patch_size", [128,128])
        self.max_images: Optional[int] = config.get("max_images")
        self.seed: int = config.get("seed", 42)
        self.fold = config.get("fold")
        self.num_folds = config.get("num_folds")
        self.verbose: bool = config.get("verbose", False)
        self.sdf_iterations: int = config.get("sdf_iterations")
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
        self.aug_cfg = _expand_aug_list(config.get("augmentation", []))

        if self.data_dim not in (2, 3):
            raise ValueError(f"data_dim must be 2 or 3, got {self.data_dim}")

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

    # Lightning/your trainer should call this at the start of every epoch
    def set_epoch(self, epoch: int):
        self._epoch = epoch

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
            print(f'{key} shape:', imgs[key].shape)
        return imgs

    def normalize_data(self, data):
        normalized_image = {}
        for key, arr in list(data.items()):
            cfg = self.norm_cfg.get(key, None)
            if cfg:
                method = cfg.get('method', None)
                params = {k: v for k, v in cfg.items() if k != 'method'}
                # print(method, params)
                normalized_image[key] = normalize_image(arr, method=method, **params)
            else:
                normalized_image[key] = arr.copy()
        return normalized_image
    
    def augment_data(self, normalized_image, sample_rng):
        if self.aug_cfg is None:
            return normalized_image
        augmented_images = {}
        for aug in self.aug_cfg:
            modalities = aug.get('modalities', None)
            if modalities is None:
                raise ValueError(f"Augmentation config for {aug} is missing 'modalities'")
            selected = {k: _to_tensor(normalized_image[k]) for k in modalities if k in normalized_image}
            augmented, meta = augment_images(selected, aug, self.data_dim, rng=sample_rng, verbose=self.verbose)
            for k_aug, v_aug in augmented.items():
                augmented_images[k_aug] = v_aug
        for key, arr in list(normalized_image.items()):
            if key not in augmented_images:
                augmented_images[key] = arr.copy()
        return augmented_images


    def _postprocess_patch(self, data: Dict[str, np.ndarray], sample_rng: np.random.Generator) -> Dict[str, np.ndarray]:
        # channel axis handling
        # 2d:(C, H, W) - 3d:(C, D, H, W)
        for k, arr in list(data.items()):
            # print('_postprocess_patch beginning0', k, data[k].ndim, data[k].shape)
            if arr.ndim == 2:
                data[k] = arr[None, ...]             # (1, H, W)
            elif arr.ndim == 3 and self.data_dim == 3:
                data[k] = arr[None, ...]             # (1, D, H, W)
            # print('_postprocess_patch beginning', k, data[k].ndim, data[k].shape)

        augment = True if self.split =='train' else False
        op = {
            "aug":  lambda d: self.augment_data(d, sample_rng) if augment else d,
            "norm": lambda d: self.normalize_data(d),
        }

        if augment:
            data = bigger_crop(data, self.patch_size, pad_mode='edge', rng=sample_rng)

        self.log_stats('before', 'step', data['label'])
        for step in self.order_ops:
            self.log_stats('before', step, data['label'])
            data = op[step](data)
            self.log_stats('after', step, data['label'])
        if self.verbose:
            print('='*50)

        if augment:
            data = center_crop(data, self.patch_size)

        data = {f"{k}_patch": _to_tensor(v) for k, v in data.items()}

        return data


    def __len__(self) -> int:
        return len(self.modality_files['image'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = worker_info.id if worker_info else 0
        # add epoch so the same idx gets a fresh seed each epoch
        seed = self.seed ^ idx ^ (self._epoch * 0x9E3779B97F4A7C15) ^ worker_seed
        rng  = np.random.default_rng(seed)

        imgs = self._load_datapoint(idx)
        if imgs is None:
            return self.__getitem__((idx + 1) % len(self))

        data = self._postprocess_patch(imgs, rng)
        
        return data

    def log_stats(self, stage: str, step: str, label: np.ndarray) -> None:
        """
        Print shape, min, and max of the label array.
        """
        if not self.verbose:
            return
        shape = label.shape
        min_val, max_val = label.min(), label.max()
        print(f"{stage:<6} | Step: {step:<10} | Shape: {shape!s:<15} | Min: {min_val:.4f} | Max: {max_val:.4f}")

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

    import yaml
    # with open('/home/ri/Desktop/Projects/Codebase/configs/dataset/main.yaml', 'w') as f_out:
        # yaml.dump(config, f_out)
    with open('./configs/dataset/AL175.yaml', 'r') as f:
    # with open('./configs/dataset/mass.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset and dataloader.
    dataset = GeneralizedDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )
    logger.info('len(dataloader): %d', len(dataloader))
    for epoch in range(10): 
        dataset.set_epoch(epoch)  
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            logger.info("Batch keys: %s", batch.keys())
            logger.info("Image shape: %s", batch["image_patch"].shape)
            logger.info("Label shape: %s", batch["label_patch"].shape)
            if config["data_dim"] == 2:
                visualize_batch_2d(batch, num_per_batch=2)
            else:
                visualize_batch_3d(batch, 2)

            # break  # Uncomment to visualize only one batch.

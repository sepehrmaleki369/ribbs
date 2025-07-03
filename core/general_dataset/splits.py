import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import KFold

def _is_junk(fname: str) -> bool:
    """Ignore index.html or config.json files (case insensitive)."""
    return fname.lower() in ("index.html", "config.json")


def _listdir_safe(folder: str) -> List[str]:
    """Safely list directory contents; return empty list if path doesn't exist."""
    try:
        return os.listdir(folder)
    except Exception:
        return []


class Split:
    """
    Unified splitter with support for 'folder', 'ratio', and 'kfold' sources.
    Adds base_modalities intersection and path validation.
    """
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.seed = cfg.get("seed", 0)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # List of modalities to intersect across. If None or single, returns all or that modality
        self.base_modalities: List[str] = cfg.get("base_modalities", [])
        if not self.base_modalities:
            # if not provided, will default to all encountered modalities later
            self.base_modalities = []

        # split_map: split_name -> modality -> list of file paths
        self.split_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self._all_mods: set = set()

        # Build raw splits from sources
        for src in cfg.get("sources", []):
            stype = src.get("type")
            if stype == "folder":
                local = self._prepare_folder_source(src)
            elif stype == "ratio":
                local = self._prepare_ratio_source(src)
            elif stype == "kfold":
                local = self._prepare_kfold_source(src)
            else:
                raise ValueError(f"Unknown source type: {stype}")

            # Merge into global map
            for split_name, mods in local.items():
                for mod, paths in mods.items():
                    self.split_map[split_name][mod].extend(paths)
                    self._all_mods.add(mod)

        # If base_modalities not specified, use all found
        if not self.base_modalities:
            self.base_modalities = sorted(self._all_mods)

    def get_split(self, split: str) -> Dict[str, List[str]]:
        """Return validated modality → file paths for a split."""
        if split not in self.split_map:
            raise KeyError(f"Split '{split}' not available; options: {list(self.split_map)}")

        # 1) Filter out junk files & ensure they exist
        valid_paths: Dict[str, List[str]] = {}
        for mod in sorted(self._all_mods):
            paths = self.split_map[split].get(mod, [])
            clean = [
                p for p in paths
                if not _is_junk(os.path.basename(p)) and os.path.exists(p)
            ]
            valid_paths[mod] = clean

        # 2) Keep only the modalities the user asked for
        for mod in list(valid_paths.keys()):
            if mod not in self.base_modalities:
                valid_paths.pop(mod)

        return valid_paths

    def _prepare_folder_source(self, src: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        base = src["path"]
        modalities = src.get("modalities", {})
        splits_cfg = src.get("splits", {})
        smap: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for split_name, subdir in splits_cfg.items():
            split_folder = os.path.join(base, subdir)
            # assume folder names for modalities match modality keys
            for mod, meta in modalities.items():
                folder = os.path.join(split_folder, mod)  # use mod name
                for fname in sorted(_listdir_safe(folder)):
                    if _is_junk(fname):
                        continue
                    full = os.path.join(folder, fname)
                    smap[split_name][mod].append(full)
        return smap

    def _collect_datapoints_for(self, src: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        root = src["path"]
        layout = src.get("layout", "flat")
        modalities = src.get("modalities", {})
        dp: Dict[str, Dict[str, str]] = defaultdict(dict)

        if layout == "folders":
            for mod, meta in modalities.items():
                sub = meta.get("folder")
                if not sub:
                    continue
                folder = os.path.join(root, sub)
                for fname in _listdir_safe(folder):
                    if _is_junk(fname):
                        continue
                    stem = os.path.splitext(fname)[0]
                    dp[stem][mod] = os.path.join(folder, fname)

        else:
            for fname in _listdir_safe(root):
                if _is_junk(fname):
                    continue
                for mod, meta in modalities.items():
                    pat = meta.get("pattern")
                    if not pat:
                        continue
                    m = re.fullmatch(pat, fname)
                    if not m:
                        continue

                    # extract stem
                    if m.groups():
                        stem = m.group(1)
                    else:
                        grp_pat = pat.replace(".*", "(.*)")
                        m2 = re.fullmatch(grp_pat, fname)
                        if m2:
                            stem = m2.group(1)
                        else:
                            stem = os.path.splitext(fname)[0]

                    dp[stem][mod] = os.path.join(root, fname)
        # ── Now prune out any stems missing a base modality ──
        if len(self.base_modalities) > 1:
            filtered = {}
            for stem, mod_map in dp.items():
                if all(mod in mod_map for mod in self.base_modalities):
                    filtered[stem] = mod_map
            dp = filtered

        return dp


    def _prepare_ratio_source(self, src: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        ratios = src.get("ratios", {})
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        dp = self._collect_datapoints_for(src)
        stems = list(dp.keys())
        random.shuffle(stems)
        n = len(stems)
        n_tr = int(n * ratios.get("train", 0))
        n_va = int(n * ratios.get("valid", 0))
        idxs = {
            "train": stems[:n_tr],
            "valid": stems[n_tr:n_tr + n_va],
            "test":  stems[n_tr + n_va:]
        }
        smap: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for split_name, group in idxs.items():
            for stem in group:
                for mod, path in dp[stem].items():
                    smap[split_name][mod].append(path)
        return smap

    def _prepare_kfold_source(self, src: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        n_folds = src.get("num_folds")
        fold_idx = src.get("fold_idx")
        tv_sources = src.get("train_valid", [])
        test_sources = src.get("test", [])

        dp_tv: Dict[str, Dict[str, str]] = {}
        for s in tv_sources:
            dp_tv.update(self._collect_datapoints_for(s))
        stems = sorted(dp_tv.keys())

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        train_ids, valid_ids = list(kf.split(stems))[fold_idx]

        smap: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for i in train_ids:
            stem = stems[i]
            for mod, path in dp_tv[stem].items():
                smap["train"][mod].append(path)
        for i in valid_ids:
            stem = stems[i]
            for mod, path in dp_tv[stem].items():
                smap["valid"][mod].append(path)

        dp_test: Dict[str, Dict[str, str]] = {}
        for s in test_sources:
            dp_test.update(self._collect_datapoints_for(s))
        for stem, mods in dp_test.items():
            for mod, path in mods.items():
                smap["test"][mod].append(path)

        return smap


def main():
    split_cfg = {
        "seed": 42,
        "sources": [
            {
                "type": "folder",
                "path": "/data/folder1",
                "layout": "folders",
                "modalities": {
                    "image": {"folder": "imgs"},
                    "label": {"folder": "lbls"}
                },
                "splits": {"train": "train_dir", "valid": "val_dir", "test": "test_dir"}
            },
            {
                "type": "ratio",
                "path": "/data/flat2",
                "layout": "flat",
                "modalities": {
                    "image": {"pattern": r".*\\.jpg$"},
                    "mask":  {"pattern": r".*\\.png$"}
                },
                "ratios": {"train": 0.7, "valid": 0.2, "test": 0.1}
            },
            {
                "type": "kfold",
                "num_folds": 5,
                "fold_idx": 0,
                "train_valid": [
                    {"path": "/data/flat_tv", "layout": "flat", "modalities": {
                        "image": {"pattern": r".*\\.npy$"},
                        "seg":   {"pattern": r".*\\.npy$"}
                    }}
                ],
                "test": [
                    {"path": "/data/flat_test", "layout": "flat", "modalities": {
                        "image": {"pattern": r".*\\.npy$"},
                        "seg":   {"pattern": r".*\\.npy$"}
                    }}
                ]
            }
        ]
    }

    splitter = Split(split_cfg)
    for split_name in ("train", "valid", "test"):
        mappings = splitter.get_split(split_name)
        print(f"=== {split_name.upper()} ===")
        for mod, files in mappings.items():
            print(f"  {mod}: {len(files)} files")

if __name__ == "__main__":
    main()

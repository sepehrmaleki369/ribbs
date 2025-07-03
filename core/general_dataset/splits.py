# core/general_dataset/split.py
# ---------------------------------------------------------------------
"""
Flexible dataset-splitting helper.

  • folder-based         (pre-made train/valid/test dirs)
  • k-fold cross-val     (plus optional dedicated test roots)
  • ratio-based shuffle  (train/valid/test percentages)

Each source folder may store modalities either
  • "folders" – one subdir per modality (sat/, map/, …)
  • "flat"    – all files mixed; regex distinguishes modalities

The stem (basename without extension) identifies a datapoint.
Returned structure: {modality: [absolute_paths …]}, aligned by stem.

Every pass through the filesystem skips any file named **index.html**
(case-insensitive).
"""
from __future__ import annotations
import os
import re
import random
import pathlib
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import KFold

# -------- helpers --------------------------------------------------- #
def _is_junk(fname: str) -> bool:
    return fname.lower() in ("index.html", "config.json")

def _listdir_safe(folder: str) -> List[str]:
    try:
        return os.listdir(folder)
    except FileNotFoundError:
        return []

# -------- main class ------------------------------------------------ #
class Split:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg        = cfg
        self.seed       = cfg.get("seed", 0)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # expose these for get_split
        src0           = cfg["source_folders"][0]
        self._all_mods = list(src0["modalities"].keys())
        self._source_root = src0["path"]
        self._is_folder_layout = src0["layout"] == "folders"
        self._modality_to_folder = {
            mod: meta.get("folder")
            for mod, meta in src0["modalities"].items()
        }

        # collect + build raw split_map
        mode = cfg["mode"]
        if mode == "folder":
            self._collect_datapoints(cfg["source_folders"])
            self.split_map = self._prepare_folder_based(cfg["source_folders"])
        elif mode == "kfold":
            self.split_map = self._prepare_kfold(cfg)
        elif mode == "ratio":
            self.split_map = self._prepare_ratio(cfg)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def get_split(self, split: str) -> Dict[str, List[str]]:
        """
        For a given split, return a dict mapping every modality to a list
        of file paths.  Missing modalities are synthesised from the
        base_modality + the source layout rules.
        """
        if split not in self.split_map:
            raise KeyError(f"Split '{split}' not found; choose from {list(self.split_map)}")

        # 1) start with whatever files were found on disk
        raw = {m: paths[:] for m, paths in self.split_map[split].items()}

        # 2) drop junk & build stem→path dicts per modality
        by_mod_stem: Dict[str, Dict[str, str]] = {}
        for mod, paths in raw.items():
            clean = [p for p in paths if not _is_junk(os.path.basename(p))]
            by_mod_stem[mod] = { pathlib.Path(p).stem: p for p in clean }

        # 3) figure out which stems to keep (those that have the base_modality)
        base = self.cfg.get("base_modality", "image")
        if base not in by_mod_stem:
            raise KeyError(f"No '{base}' files found in split '{split}'")
        stems = sorted(by_mod_stem[base].keys())

        # 4) for each stem, collect real or synthed paths for all modalities
        filemap: Dict[str, List[str]] = {m: [] for m in self._all_mods}
        for stem in stems:
            base_path = by_mod_stem[base][stem]
            base_dir  = os.path.dirname(base_path)
            for mod in self._all_mods:
                if stem in by_mod_stem.get(mod, {}):
                    # already on disk
                    filemap[mod].append(by_mod_stem[mod][stem])
                else:
                    # synthesize
                    if self._is_folder_layout:
                        split_root = os.path.join(self._source_root, split)
                        mod_dir    = os.path.join(split_root, self._modality_to_folder[mod])
                        fname      = f"{stem}_{mod}.npy"
                    else:
                        mod_dir = base_dir
                        if mod in ("image","label"):
                            # look for any matching existing file
                            candidates = [
                                p for p in os.listdir(mod_dir)
                                if pathlib.Path(p).stem == stem and mod in p
                            ]
                            fname = candidates[0] if candidates else f"{stem}_{mod}"
                        else:
                            fname = f"{stem}_{mod}.npy"
                    filemap[mod].append(os.path.join(mod_dir, fname))

        return filemap

    def iter_datapoints(self):
        for stem, mods in self._all_datapoints.items():
            yield stem, mods

    # -------- split-builders ---------------------------------------- #
    def _prepare_folder_based(self, src_cfg):
        smap = defaultdict(lambda: defaultdict(list))
        for sf in src_cfg:
            for split, subdir in sf["splits"].items():
                files = self._scan_folder(os.path.join(sf["path"], subdir), sf)
                for m, plist in files.items():
                    smap[split][m].extend(plist)
        return smap

    def _prepare_kfold(self, cfg):
        tr_cfg, test_cfg = cfg["train_valid_folders"], cfg["test_folders"]
        n_fold, idx      = cfg["num_folds"], cfg["fold_idx"]
        dp = self._collect_datapoints(tr_cfg)
        stems = sorted(dp)
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=self.seed)
        tr_ids, val_ids = list(kf.split(stems))[idx]

        smap = {s: defaultdict(list) for s in ("train","valid","test")}
        for i in tr_ids:
            for m,p in dp[stems[i]].items(): smap["train"][m].append(p)
        for i in val_ids:
            for m,p in dp[stems[i]].items(): smap["valid"][m].append(p)
        for sf in test_cfg:
            files = self._scan_folder(sf["path"], sf, ignore_splits=True)
            for m,plist in files.items(): smap["test"][m].extend(plist)
        return smap

    def _prepare_ratio(self, cfg):
        folders, ratios = cfg["ratio_folders"], cfg["ratios"]
        if abs(sum(ratios.values()) - 1) > 1e-6:
            raise ValueError("ratios must sum to 1")
        dp = self._collect_datapoints(folders)
        stems = list(dp); random.shuffle(stems)
        n   = len(stems)
        n_tr = int(n * ratios["train"])
        n_va = int(n * ratios["valid"])

        idx = {
            "train": stems[:n_tr],
            "valid": stems[n_tr:n_tr+n_va],
            "test":  stems[n_tr+n_va:]
        }
        smap = {s: defaultdict(list) for s in idx}
        for split, slist in idx.items():
            for s in slist:
                for m,p in dp[s].items(): smap[split][m].append(p)
        return smap

    # -------- filesystem utils -------------------------------------- #
    def _collect_datapoints(self, folders_cfg):
        local = defaultdict(dict)
        for sf in folders_cfg:
            root, layout = sf["path"], sf["layout"]
            if layout == "folders":
                for mod, meta in sf["modalities"].items():
                    mdir = os.path.join(root, meta["folder"])
                    for fname in _listdir_safe(mdir):
                        if _is_junk(fname): continue
                        stem = pathlib.Path(fname).stem
                        path = os.path.join(mdir, fname)
                        local[stem][mod] = path
                        self._all_datapoints[stem][mod] = path
            else:
                for fname in _listdir_safe(root):
                    if _is_junk(fname): continue
                    for mod, meta in sf["modalities"].items():
                        if re.fullmatch(meta["pattern"], fname):
                            stem = pathlib.Path(fname).stem
                            path = os.path.join(root, fname)
                            local[stem][mod] = path
                            self._all_datapoints[stem][mod] = path
        return local

    def _scan_folder(self, folder, sf_cfg, ignore_splits=False):
        res = defaultdict(list)
        if sf_cfg["layout"] == "folders":
            for mod, meta in sf_cfg["modalities"].items():
                mdir = folder if ignore_splits else os.path.join(folder, meta["folder"])
                for fname in _listdir_safe(mdir):
                    if _is_junk(fname): continue
                    res[mod].append(os.path.join(mdir, fname))
        else:
            for fname in _listdir_safe(folder):
                if _is_junk(fname): continue
                for mod, meta in sf_cfg["modalities"].items():
                    if re.fullmatch(meta["pattern"], fname):
                        res[mod].append(os.path.join(folder, fname))
        return res

# ------------------------------------------------------------------ #
# quick demo                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    demo_cfg = {
        "seed": 1,
        "mode": "folder",
        "base_modality": "image",
        "source_folders": [
            {
                "path": "/path/to/dataset",
                "layout": "folders",
                "modalities": {
                    "image": {"folder": "sat"},
                    "label": {"folder": "label"},
                    "sdf":   {"folder": "sdf"},
                },
                "splits": {"train": "train", "valid": "valid", "test": "test"}
            }
        ]
    }
    splitter = Split(demo_cfg)
    for s in ("train", "valid", "test"):
        filemap = splitter.get_split(s)
        print(f"{s}:")
        for mod, paths in filemap.items():
            print(f"  {mod}: {len(paths)} samples")

# ------------------------------------------------------------------ #
# sanity check                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    cfg_demo = {
        "seed": 1,
        "base_modality": 'image',
        "mode": "ratio",
        "ratios": {"train": .8, "valid": .1, "test": .1},
        "ratio_folders": [
            {
                "path": "dummy_dataset",
                "layout": "flat",
                "modalities": {
                    "image": {"pattern": r".*_sat\.tif"},
                    "label": {"pattern": r".*_map\.png"}
                }
            }
        ]
    }
    sp = Split(cfg_demo)
    for s in ("train", "valid", "test"):
        print(s, {m: len(v) for m, v in sp.get_split(s).items()})

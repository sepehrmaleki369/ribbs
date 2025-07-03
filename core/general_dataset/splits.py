import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Any, Tuple
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

def filename_from_pattern(pattern: str, stem: str) -> str:
    # 1) strip regex anchors
    if pattern.startswith("^"):
        pattern = pattern[1:]
    if pattern.endswith("$"):
        pattern = pattern[:-1]

    # 2) split on the literal "(.*)"
    parts = pattern.split("(.*)")
    if len(parts) != 2:
        raise ValueError(f"Pattern must contain exactly one '(.*)' slot: {pattern!r}")
    prefix, suffix = parts

    # 3) un-escape any escaped chars (e.g. "\." → ".", "\(" → "(", etc.)
    unescape = lambda s: re.sub(r"\\(.)", r"\1", s)
    prefix = unescape(prefix)
    suffix = unescape(suffix)

    # 4) re-join
    return f"{prefix}{stem}{suffix}"

def _filter_complete(records, required_modalities):
    by_group = defaultdict(list)
    for rec in records:
        key = (rec["split"], rec["stem"])
        by_group[key].append(rec)

    complete = []
    for (split, stem), recs in by_group.items():
        mods = {r["modality"] for r in recs}
        if mods >= set(required_modalities):
            complete.extend(recs)
    return complete

def _filter_complete_no_split(records: List[Dict[str, str]], required_modalities: List[str]) -> List[Dict[str, str]]:
    """
    Given a flat list of records without split information,
    drop any stems missing one of the required_modalities.
    """
    # Group records by stem
    by_stem: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        stem = rec["stem"]
        by_stem[stem].append(rec)

    # Keep only those groups whose modalities cover the required set
    req_set = set(required_modalities)
    complete: List[Dict[str, str]] = []
    for stem, recs in by_stem.items():
        mods = {r["modality"] for r in recs}
        if mods >= req_set:
            complete.extend(recs)

    return complete

def _pivot_views(records):
    st_md_sp = defaultdict(lambda: defaultdict(dict))
    md_st_sp = defaultdict(lambda: defaultdict(dict))
    st_sp_md = defaultdict(lambda: defaultdict(dict))
    sp_st_md = defaultdict(lambda: defaultdict(dict))

    for r in records:
        s, m, t, p = r["split"], r["modality"], r["stem"], r["path"]
        st_md_sp[t][m][s] = p
        md_st_sp[m][t][s] = p
        st_sp_md[t][s][m] = p
        sp_st_md[s][t][m] = p

    return [st_md_sp, md_st_sp, st_sp_md, sp_st_md]

def _pivot_views_no_split(records):
    st_md = defaultdict(lambda: defaultdict(dict))
    md_st = defaultdict(lambda: defaultdict(dict))

    for r in records:
        m, t, p = r["modality"], r["stem"], r["path"]
        st_md[t][m] = p
        md_st[m][t] = p

    return [st_md, md_st]

def _collect_datapoints_from_source(src: Dict[str, Any], base_modalities, rng):
    root       = src["path"]
    layout     = src.get("layout", "flat")
    modalities = src.get("modalities", {})
    splits     = src.get("splits", {})

    base_records = []
    if layout == "folders":
        for split, subfolder in splits.items():
            split_dir = os.path.join(root, subfolder)
            for mod, meta in modalities.items():
                if mod not in base_modalities:
                    continue
                subfolder = meta.get("folder")
                if not subfolder:
                    continue
                folder = os.path.join(split_dir, subfolder)
                for fname in _listdir_safe(folder):
                    if _is_junk(fname):
                        continue
                    stem = os.path.splitext(fname)[0]
                    path = os.path.join(folder, fname)
                    base_records.append({
                        "split":    split,
                        "modality": mod,
                        "stem":     stem,
                        "path":     path,
                    })

        # 2) drop any (split,stem) groups missing a base modality
        base_records = _filter_complete(base_records, base_modalities)
        # extract all the valid stems per split
        stems_by_split = defaultdict(set)
        for rec in base_records:
            stems_by_split[rec["split"]].add(rec["stem"])

        # 3) start your final list with the complete base records
        records = list(base_records)

        # 4) now add any non-base modalities, if the file exists  
        for split, valid_stems in stems_by_split.items():
            split_dir = os.path.join(root, split)
            for mod, meta in modalities.items():
                if mod in base_modalities:
                    continue
                subfolder = meta.get("folder")
                if not subfolder:
                    continue
                folder = os.path.join(split_dir, subfolder)
                for stem in valid_stems:
                    fname = f"{stem}_{mod}.npy"
                    records.append({
                        "split":    split,
                        "modality": mod,
                        "stem":     stem,
                        "path":     os.path.join(folder, fname),
                    })

        # print(_pivot_views(records)[-1])
        
    else:
        base_records = []
        for fname in _listdir_safe(root):
            if _is_junk(fname):
                continue
            for mod, meta in modalities.items():
                if mod not in base_modalities:
                    continue
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
                
                base_records.append({
                        "modality": mod,
                        "stem":     stem,
                        "path":     os.path.join(root, fname),
                    })
        

        # Filter out incomplete groups
        complete_base = _filter_complete_no_split(base_records, base_modalities)
        valid_stems = {rec["stem"] for rec in complete_base}

        # Add complete base records
        records = list(complete_base)
        # print(_pivot_views_no_split(records)[0])
        # raise

        # Now append non-base modalities
        for mod, meta in modalities.items():
            if mod in base_modalities:
                continue
            pat = meta.get("pattern")
            if not pat:
                continue
            for stem in valid_stems:
                escaped_stem = re.escape(stem)
                full_pat = pat.replace("(.*)", f"({escaped_stem})")
                fname   = filename_from_pattern(pat, stem)
                # print(mod, stem)
                # print(fname)
                records.append({
                    "modality": mod,
                    "stem":     stem,
                    "path":     os.path.join(root, fname),
                })
        # print(_pivot_views_no_split(records)[0])
        records = split_records(src, records, rng)
    
    return records

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

        self.base_modalities = cfg.get("base_modalities", [])
        
        # Make the RNG deterministic but *local* to this instance
        self._rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Filled lazily by _build_splits()
        self._splits_built = False
        self._split2mod2files: Dict[str, Dict[str, List[str]]] = {}

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def get_split(self, split: str) -> Dict[str, List[str]]:
        """
        Args
        ----
        split : str
            One of "train", "valid", "test".

        Returns
        -------
        Dict[str, List[str]]
            { modality : [file paths] }  (lists are *already sorted*).
        """
        if split not in ("train", "valid", "test"):
            raise ValueError("split must be 'train', 'valid' or 'test'")

        if not self._splits_built:
            self._build_splits()

        # `.get` so an empty dict is returned if this split doesn't exist
        return self._split2mod2files.get(split, {})

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _build_splits(self) -> None:
        """
        One-shot scan of every source in the config → populate
        self._split2mod2files.
        """
        all_records: List[Dict[str, str]] = []

        # 1) collect and (if needed) split each source
        for src in self.cfg.get("sources", []):
            # recs = _collect_datapoints_from_source(src, self.base_modalities)
            recs = self._collect_datapoints_for(src)
            all_records.extend(recs)

        # 2) bucket by split / modality
        split2mod2files = defaultdict(lambda: defaultdict(list))
        for rec in all_records:
            sp, mod, path = rec["split"], rec["modality"], rec["path"]
            split2mod2files[sp][mod].append(path)

        # 3) deterministic order: sort the lists so paired modalities stay aligned
        for sp in split2mod2files:
            for mod in split2mod2files[sp]:
                split2mod2files[sp][mod].sort()

        # ---- print a summary of each split ----
        for sp, mod2files in split2mod2files.items():
            # count “stems” via the first base modality
            if self.base_modalities:
                base = self.base_modalities[0]
                stem_count = len(mod2files.get(base, []))
            else:
                stem_count = sum(len(v) for v in mod2files.values()) // max(1, len(mod2files))
            print(f"→ Split '{sp}': {stem_count} stems")
            for mod, files in mod2files.items():
                print(f"     {mod:8s}: {len(files)} files")
            print()
        # ---- end summary ----

        self._split2mod2files = split2mod2files
        self._splits_built    = True

    def _collect_datapoints_for(self, src: Dict[str, Any]) -> List[Dict[str, str]]:
        records = _collect_datapoints_from_source(src, self.base_modalities, self._rng)
        if src.get('type')=='kfold':
            test_src = src.get('test_source')
            test_records = _collect_datapoints_from_source(test_src, self.base_modalities, self._rng)  
            records.extend([rec for rec in test_records if rec['split']=='test'])
        return records
    
def split_records(src, records, rng):
    """
    Assigns a 'split' field to each record in `records` based on the split strategy in `src`.

    - For 'ratio': groups by stem, shuffles, and slices by the provided ratios.
    - For 'kfold': performs K-fold cross-validation on stems, using fold_idx as the held-out fold.
    """
    split_type = src.get('type')
    if split_type == 'folder':  
        return records
    
    # RATIO-BASED SPLIT
    if split_type == 'ratio':
        # Collect unique stems
        stems = sorted({r['stem'] for r in records})
        # Shuffle with seed if provided
        # seed = src.get('seed', None)
        # if seed is not None:
        #     random.Random(seed).shuffle(stems)
        # else:
        #     random.shuffle(stems)
        rng.shuffle(stems)

        ratios = src.get('ratios', {})
        train_ratio = ratios.get('train', 0)
        valid_ratio = ratios.get('valid', 0)
        # test_ratio implied
        n = len(stems)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        # Ensure all accounted for
        n_test = n - n_train - n_valid

        train_stems = set(stems[:n_train])
        valid_stems = set(stems[n_train:n_train + n_valid])
        test_stems  = set(stems[n_train + n_valid:])

        # Assign splits
        for rec in records:
            s = rec['stem']
            if s in train_stems:
                rec['split'] = 'train'
            elif s in valid_stems:
                rec['split'] = 'valid'
            else:
                rec['split'] = 'test'
        return records

    # K-FOLD SPLIT
    elif split_type == 'kfold':
        num_folds = src.get('num_folds')
        fold_idx  = src.get('fold_idx', 0)
        # Collect unique stems
        stems = sorted({r['stem'] for r in records})
        # Prepare KFold
        kf = KFold(n_splits=num_folds,
                   shuffle=True,
                   random_state=src.get('seed', None))
        # Find the train/valid split for the requested fold
        for idx, (train_idx, valid_idx) in enumerate(kf.split(stems)):
            if idx == fold_idx:
                train_stems = {stems[i] for i in train_idx}
                valid_stems = {stems[i] for i in valid_idx}
                break

        # Assign splits
        for rec in records:
            if rec['stem'] in train_stems:
                rec['split'] = 'train'
            elif rec['stem'] in valid_stems:
                rec['split'] = 'valid'
                
        return records

    else:
        raise ValueError(f"Unsupported split type: {split_type}")


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
                "path": "/data/flat_tv",
                "layout": "flat",
                "modalities": {
                    "image": {"pattern": r".*\.npy$"},
                    "seg":   {"pattern": r".*\.npy$"}
                },
                "test_source":{
                    # this can only be type ratio 
                }
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

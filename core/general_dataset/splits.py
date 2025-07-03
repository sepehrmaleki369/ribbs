import numpy as np
from sklearn.model_selection import KFold
from typing import Any, Dict, List, Optional
from core.general_dataset.logger import logger


























def kfold_split(files: List[str], num_folds: int, fold: int, seed: int):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(files))
    return splits[fold]

def ratio_split(files: List[str], ratios: Dict[str,float], split: str, seed: int):
    np.random.seed(seed)
    idx = np.random.permutation(len(files))
    t_cnt = int(len(files)*ratios['train'])
    v_cnt = int(len(files)*ratios['valid'])
    if split=='train': sel = idx[:t_cnt]
    elif split=='valid': sel = idx[t_cnt:t_cnt+v_cnt]
    else: sel = idx[t_cnt+v_cnt:]
    return sel.tolist()

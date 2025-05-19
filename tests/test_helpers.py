"""
Unit tests for critical helper functions in the segmentation framework.
Run with: pytest -xvs test_helpers.py
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the functions to test
from core.general_dataset import compute_distance_map, compute_sdf, custom_collate_fn
from core.utils import (
    noCrops,
    noCropsPerDim,
    cropInds,
    coord,
    coords,
    cropCoords,
    process_in_chuncks,
)


class TestDistanceMap:
    """Tests for compute_distance_map function"""

    def test_basic_functionality(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        distance = compute_distance_map(mask, None)

        assert distance[2, 2] == 0
        for i, j in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            assert 2.8 < distance[i, j] < 2.9
        for i, j in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            assert distance[i, j] == 1

    def test_thresholding(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1
        distance = compute_distance_map(mask, 2.0)

        assert np.max(distance) <= 2.0
        assert distance[3, 3] == 0
        assert distance[2, 3] == 1
        assert distance[4, 3] == 1

    def test_binary_formats(self):
        mask_01 = np.zeros((5, 5), dtype=np.uint8); mask_01[2, 2] = 1
        mask_0255 = np.zeros((5, 5), dtype=np.uint8); mask_0255[2, 2] = 255
        d1 = compute_distance_map(mask_01, None)
        d2 = compute_distance_map(mask_0255, None)
        assert np.array_equal(d1, d2)

    def test_empty_mask(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        distance = compute_distance_map(mask, None)
        assert np.all(distance > 0)

    def test_full_mask(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        distance = compute_distance_map(mask, None)
        assert np.all(distance == 0)


class TestSignedDistanceFunction:
    """Tests for compute_sdf function"""

    def test_basic_functionality(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1
        sdf = compute_sdf(mask, sdf_iterations=1, sdf_thresholds=None)

        # Inside (where mask==1) should be negative
        assert sdf[3, 3] < 0
        # Far away should be positive
        assert sdf[0, 0] > 0

    def test_iterations_parameter(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1

        sdf1 = compute_sdf(mask, sdf_iterations=1, sdf_thresholds=None)
        sdf3 = compute_sdf(mask, sdf_iterations=3, sdf_thresholds=None)

        neg1 = np.sum(sdf1 < 0)
        neg3 = np.sum(sdf3 < 0)
        # More iterations should not shrink the "inside"—allow equal or larger
        assert neg3 >= neg1

    def test_thresholds_parameter(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1

        sdf = compute_sdf(mask, sdf_iterations=1, sdf_thresholds=[-2, 2])
        assert np.all(sdf >= -2)
        assert np.all(sdf <= 2)

    def test_binary_formats(self):
        mask_01 = np.zeros((5, 5), dtype=np.uint8); mask_01[2, 2] = 1
        mask_0255 = np.zeros((5, 5), dtype=np.uint8); mask_0255[2, 2] = 255
        s1 = compute_sdf(mask_01, sdf_iterations=1, sdf_thresholds=None)
        s2 = compute_sdf(mask_0255, sdf_iterations=1, sdf_thresholds=None)
        assert np.array_equal(s1, s2)


class TestCropFunctions:
    """Tests for the crop‐related utility functions"""

    def test_noCrops_basic(self):
        assert noCrops([100, 100], [50, 50], [5, 5], startDim=0) == 9

    def test_noCrops_tiny_image(self):
        assert noCrops([10, 10], [10, 10], [3, 3], startDim=0) == 1

    def test_noCropsPerDim(self):
        per, cum = noCropsPerDim([100, 200], [50, 50], [5, 5], startDim=0)
        assert per == [3, 5]
        assert cum == [15, 5, 1]

    def test_cropInds(self):
        cum = [12, 3, 1]
        assert cropInds(0, cum) == [0, 0]
        assert cropInds(3, cum) == [1, 0]
        assert cropInds(11, cum) == [3, 2]

    def test_coord(self):
        c, v = coord(2, 30, 5, 100)
        assert (c.start, c.stop) == (40, 70)
        assert (v.start, v.stop) == (5, 25)

        c, v = coord(4, 30, 5, 100)
        assert (c.start, c.stop) == (70, 100)
        # when hitting edge, valid region is trimmed (start=15) to avoid going out of bounds
        assert (v.start, v.stop) == (15, 30)

    def test_coords(self):
        cc, vc = coords([1, 2], [30, 30], [5, 5], [100, 100], 0)
        assert (cc[0].start, cc[0].stop) == (20, 50)
        assert (cc[1].start, cc[1].stop) == (40, 70)
        assert (vc[0].start, vc[0].stop) == (5, 25)
        assert (vc[1].start, vc[1].stop) == (5, 25)

    def test_cropCoords(self):
        cc, vc = cropCoords(7, [30, 40], [5, 5], [100, 200], 0)
        for sl in [*cc, *vc]:
            assert isinstance(sl, slice)
        assert 0 <= cc[0].start < cc[0].stop <= 100
        assert 0 <= cc[1].start < cc[1].stop <= 200
        assert 0 <= vc[0].start < vc[0].stop <= 30
        assert 0 <= vc[1].start < vc[1].stop <= 40


class TestProcessInChunks:
    """Tests for process_in_chuncks"""

    def test_basic_processing(self):
        inp = torch.ones((1, 3, 10, 10))
        out = torch.zeros((1, 1, 10, 10))
        def fn(x): return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        res = process_in_chuncks(inp, out, fn, [5, 5], [1, 1])
        assert torch.all(res == 1)

    def test_chunking_logic(self):
        inp = torch.zeros((1, 3, 20, 20))
        out = torch.zeros((1, 1, 20, 20))
        def fn(x):
            b, c, h, w = x.shape
            t = torch.zeros((b, 1, h, w))
            for i in range(h):
                for j in range(w):
                    t[0, 0, i, j] = i + j
            return t
        res = process_in_chuncks(inp, out, fn, [10, 10], [2, 2])
        assert res[0, 0, 0, 0] == 0
        assert res[0, 0, 6, 0] == 6

    def test_shape_handling(self):
        inp = torch.ones((1, 3, 10, 10))
        out = torch.zeros((1, 1, 10, 10))
        def fn(x): return torch.ones((x.shape[0], x.shape[2], x.shape[3]))
        res = process_in_chuncks(inp, out, fn, [5, 5], [1, 1])
        assert torch.all(res == 1)

    def test_margin_handling(self):
        inp = torch.zeros((1, 1, 20, 20))
        for i in range(20):
            for j in range(20):
                inp[0, 0, i, j] = i * 100 + j
        out = torch.zeros((1, 1, 20, 20))
        def fn(x): return x
        res = process_in_chuncks(inp, out, fn, [10, 10], [2, 2])
        assert torch.all(res == inp)


class TestDataSplitting:
    """Tests for data splitting logic (ratio and k-fold)"""

    @pytest.fixture
    def mock_file_structure(self, tmp_path):
        """
        Create a mock dataset structure for testing:
        
        dataset/
          source/
            sat/  (images)
            map/  (labels)
        """
        root = tmp_path / "dataset"
        img_dir = root / "source" / "sat"
        lbl_dir = root / "source" / "map"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 10 dummy .tif files in each
        for i in range(10):
            (img_dir / f"img_{i}.tif").write_text("dummy")
            (lbl_dir / f"img_{i}.tif").write_text("dummy")

        return root

    def test_ratio_splitting(self, mock_file_structure, monkeypatch):
        """Test ratio-based splitting logic"""
        from core.general_dataset import GeneralizedDataset

        # Prevent any random shuffle
        monkeypatch.setattr(np.random, "shuffle", lambda x: x)

        base_cfg = {
            "root_dir": str(mock_file_structure),
            "use_splitting": True,
            "source_folder": "source",
            "split_ratios": {"train": 0.6, "valid": 0.2, "test": 0.2},
            "modalities": {"image": "sat", "label": "map"},
        }

        # Create each split explicitly
        train_ds = GeneralizedDataset({**base_cfg, "split": "train"})
        valid_ds = GeneralizedDataset({**base_cfg, "split": "valid"})
        test_ds  = GeneralizedDataset({**base_cfg, "split": "test"})

        # Should be 6,2,2 images respectively
        assert len(train_ds.modality_files["image"]) == 6
        assert len(valid_ds.modality_files["image"]) == 2
        assert len(test_ds.modality_files["image"])  == 2

        # No overlap
        t = set(train_ds.modality_files["image"])
        v = set(valid_ds.modality_files["image"])
        s = set(test_ds.modality_files["image"])
        assert t.isdisjoint(v)
        assert t.isdisjoint(s)
        assert v.isdisjoint(s)

    def test_kfold_splitting(self, mock_file_structure, monkeypatch):
        """Test k-fold splitting logic"""
        from core.general_dataset import GeneralizedDataset

        # Mock KFold to return a single fixed split (first 8 train, last 2 valid)
        class MockKFold:
            def __init__(self, n_splits, shuffle, random_state):
                pass
            def split(self, X):
                return [(np.arange(8), np.arange(8, 10))]

        monkeypatch.setattr("sklearn.model_selection.KFold", MockKFold)

        base_cfg = {
            "root_dir": str(mock_file_structure),
            "use_splitting": True,
            "split_mode": "kfold",
            "num_folds": 5,
            "source_folder": "source",
            "modalities": {"image": "sat", "label": "map"},
        }

        # Train fold 0
        tr_cfg = {**base_cfg, "split": "train", "fold": 0}
        train_ds = GeneralizedDataset(tr_cfg)
        assert len(train_ds.modality_files["image"]) == 8

        # Valid fold 0
        va_cfg = {**base_cfg, "split": "valid", "fold": 0}
        valid_ds = GeneralizedDataset(va_cfg)
        assert len(valid_ds.modality_files["image"]) == 2


class TestCustomCollate:
    """Tests for custom_collate_fn"""

    def test_basic(self):
        batch = [
            {"image": torch.ones(3,10,10), "label": torch.zeros(1,10,10)},
            {"image": torch.ones(3,10,10), "label": torch.zeros(1,10,10)},
        ]
        c = custom_collate_fn(batch)
        assert c["image"].shape == (2,3,10,10)
        assert c["label"].shape == (2,1,10,10)

    def test_mixed_types(self):
        batch = [
            {"image": torch.ones(3,10,10), "meta": {"id":1}},
            {"image": torch.ones(3,10,10), "meta": {"id":2}},
        ]
        c = custom_collate_fn(batch)
        assert isinstance(c["meta"], list)
        assert c["meta"][0]["id"] == 1

    def test_filter_none(self):
        def imp(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return {"image": torch.zeros(0,3,10,10)}
            return custom_collate_fn(batch)
        b = [None, {"image": torch.ones(3,10,10)}]
        c1 = imp(b)
        assert c1["image"].shape[0] == 1
        c2 = imp([None,None])
        assert c2["image"].shape[0] == 0

import numpy as np
import pytest
from core.general_dataset.augments import flip_h, flip_v, flip_d, rotate_, extract_condition_augmentations

def make_img2d():
    return np.arange(100).reshape(10,10).astype(float)

def make_img3d():
    # 1 channel, 4 × 10×10 volume
    return np.arange(400).reshape(1,4,10,10).astype(float)

def test_flip_2d():
    im = make_img2d()
    assert np.all(flip_h(flip_h(im)) == im)
    assert np.all(flip_v(flip_v(im)) == im)

def test_flip_3d():
    vol = make_img3d()
    assert np.all(flip_d(flip_d(vol)) == vol)

def test_rotate_2d_shape():
    im = make_img2d()
    meta = {'x': 2, 'y': 3, 'angle': 30}
    out = rotate_(im, meta, patch_size_xy=5, patch_size_z=1, data_dim=2)
    assert out.shape == (5,5)

def test_rotate_3d_shape():
    vol = make_img3d()
    meta = {'x':1,'y':1,'z':1,'angle': 45}
    out = rotate_(vol, meta, patch_size_xy=4, patch_size_z=2, data_dim=3)
    # Expect shape C×Z×XY×XY → (1,2,4,4)
    assert out.shape == (1,2,4,4)

import os
import numpy as np
from tqdm import tqdm
import warnings

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    RASTERIO_AVAILABLE = True
except ImportError:
    from PIL import Image
    RASTERIO_AVAILABLE = False


def read_image(path):
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
    else:
        img = np.array(Image.open(path).convert("RGB"))
    return img


def read_label(path):
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            lbl = src.read(1)  # read first band as label
    else:
        lbl = np.array(Image.open(path))
    return lbl


def save_label(label_array, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if RASTERIO_AVAILABLE:
        height, width = label_array.shape
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': str(label_array.dtype),
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(label_array, 1)
    else:
        from PIL import Image
        Image.fromarray(label_array).save(out_path)


def clean_labels(image_dir, label_dir, output_dir, window_size=8, white_threshold=250):
    """
    For each label pixel block (window_size x window_size),
    if the corresponding image block is 'white' (above white_threshold),
    set those label pixels to 0.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather image and label files by base name, ignoring extension differences
    image_files = {os.path.splitext(f)[0]: os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))}

    label_files = {os.path.splitext(f)[0]: os.path.join(label_dir, f)
                   for f in os.listdir(label_dir)
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))}

    common_keys = sorted(set(image_files.keys()) & set(label_files.keys()))

    for key in tqdm(common_keys, desc=f"Cleaning labels from {image_dir}"):
        image_path = image_files[key]
        label_path = label_files[key]
        output_path = os.path.join(output_dir, f"{key}.tif")  # force .tif output

        try:
            image = read_image(image_path)
            label = read_label(label_path)
        except Exception as e:
            print(f"Failed to load {key}: {e}")
            continue

        # If you have an alpha channel, ignore it
        if image.shape[-1] == 4:
            image = image[..., :3]

        H, W = image.shape[:2]
        cleaned = label.copy()

        for y in range(0, H, window_size):
            for x in range(0, W, window_size):
                y1 = min(y + window_size, H)
                x1 = min(x + window_size, W)
                patch = image[y:y1, x:x1]  # shape ~ (8, 8, 3)

                # is_every_pixel_white? => (pixel_value > white_threshold) in all channels
                if np.all(patch > white_threshold):
                    cleaned[y:y1, x:x1] = 0

        save_label(cleaned, output_path)

    print(f"Finished cleaning {len(common_keys)} matched label-image pairs.")


if __name__ == "__main__":
    # Example usage:
    clean_labels(
        image_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/train/sat",
        label_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/train/map",
        output_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/train/label",
        window_size=8,
        white_threshold=250  # can tune to ~240-255
    )

    clean_labels(
        image_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/valid/sat",
        label_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/valid/map",
        output_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/valid/label",
        window_size=8,
        white_threshold=250
    )

    clean_labels(
        image_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/test/sat",
        label_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/test/map",
        output_dir="/home/ri/Desktop/Projects/Datasets/Mass_Roads/dataset/test/label",
        window_size=8,
        white_threshold=250
    )

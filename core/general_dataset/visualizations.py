from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch_2d(batch: Dict[str, Any], num_per_batch: Optional[int] = None) -> None:
    """
    Visualizes patches in a batch: image, label, distance, and SDF (if available).

    Args:
        batch (Dict[str, Any]): Dictionary containing batched patches.
        num_per_batch (Optional[int]): Maximum number of patches to visualize.
    """
    print('batch["image_patch"].shape:', batch["image_patch"].shape)
    num_to_plot = batch["image_patch"].shape[0]
    if num_per_batch:
        num_to_plot = min(num_to_plot, num_per_batch)
    for i in range(num_to_plot):
        sample_image = batch["image_patch"][i].numpy()
        if sample_image.shape[0] == 3:  # CHW to HWC
            sample_image = sample_image.transpose(1, 2, 0)
        elif sample_image.shape[0] == 1:
            sample_image = sample_image[0]  # grayscale
        else:
            sample_image = sample_image.transpose(1, 2, 0)

        sample_label = np.squeeze(batch["label_patch"][i].numpy())
        sample_distance = batch["distance_patch"][i].numpy() if "distance_patch" in batch else None
        sample_sdf = batch["sdf_patch"][i].numpy() if "sdf_patch" in batch else None

        print(f'Patch {i}')
        print('  image:', sample_image.min(), sample_image.max())
        print('  label:', sample_label.min(), sample_label.max())
        if sample_distance is not None:
            print('  distance:', sample_distance.min(), sample_distance.max())
        if sample_sdf is not None:
            print('  sdf:', sample_sdf.min(), sample_sdf.max())

        num_subplots = 3 + (1 if sample_sdf is not None else 0)
        fig, axs = plt.subplots(1, num_subplots, figsize=(12, 4))
        axs[0].imshow(sample_image, cmap='gray' if sample_image.ndim == 2 else None)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(sample_label, cmap='gray')
        axs[1].set_title("Label")
        axs[1].axis("off")
        if sample_distance is not None:
            axs[2].imshow(sample_distance[0], cmap='gray')
            axs[2].set_title("Distance")
            axs[2].axis("off")
        else:
            axs[2].text(0.5, 0.5, "No Distance", ha='center', va='center')
            axs[2].axis("off")
        if sample_sdf is not None:
            axs[3].imshow(sample_sdf[0], cmap='coolwarm')
            axs[3].set_title("SDF")
            axs[3].axis("off")
        plt.tight_layout()
        plt.show()


def visualize_batch_3d(
    batch: Dict[str, Any],
    projection: str = "max",
    num_per_batch: Optional[int] = None
) -> None:
    """
    Visualizes 3D patches in a batch by projecting along the Z axis.
    
    Args:
        batch: dict with keys image_patch, label_patch, etc., each a Tensor [B,C,Z,H,W] or [B,Z,H,W]
        projection: one of "max", "min", "mean"
        num_per_batch: how many samples to plot
    """
    assert projection in ("max","min","mean"), "projection must be max, min, or mean"
    
    # which modalities?
    print('batch', batch.keys())
    mods = [k.replace("_patch","") for k in batch if k.endswith("_patch")]
    print('batch["image_patch"].shape:', batch["image_patch"].shape)
    num_to_plot = batch["image_patch"].shape[0]
    if num_per_batch:
        num_to_plot = min(num_to_plot, num_per_batch)
    
    for i in range(num_to_plot):
        # gather each modality’s volume
        vols = {}
        for mod in mods:
            arr = batch[f"{mod}_patch"][i].numpy()
            # arr is either (Z,H,W) or (C,Z,H,W)
            if arr.ndim == 4:
                # collapse channels by max
                arr = arr.max(axis=0)
            vols[mod] = arr  # now Z×H×W
        
        # set up subplot grid: rows=modalities, cols=1
        nrows = len(mods)
        ncols = 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        project = {
            "max": np.max,
            "min": np.min,
            "mean": np.mean
        }[projection]
        print(f'Patch {i} ({projection}-projection)')
        for row, mod in enumerate(mods):
            vol = vols[mod]
            # Compute projections along axes:
            #   XY: collapse Z axis -> (H, W)
            #   XZ: collapse Y axis -> (Z, W)
            #   YZ: collapse X axis -> (Z, H) then transpose for display (H, Z)
            proj_xy = project(vol, axis=0)
            proj_xz = project(vol, axis=1)
            proj_yz = project(vol, axis=2).T

            projs = {"XY": proj_xy, "XZ": proj_xz, "YZ": proj_yz}

            for col, (name, proj) in enumerate(projs.items()):
                ax = axs[row, col] if nrows > 1 else axs[col]
                ax.imshow(proj, cmap='gray')
                ax.set_title(f"{mod} - {name} view")
                ax.axis("off")

                # Print projection stats
                print(f'  {mod} {name}: min={proj.min():.3f}, max={proj.max():.3f}, mean={proj.mean():.3f}')

        plt.tight_layout()
        plt.show()
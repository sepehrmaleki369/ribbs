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
    slice_dim: int = 2
) -> None:
    """
    Visualizes 3D patches in a batch by projecting along the Z axis.
    
    Args:
        batch: dict with keys image_patch, label_patch, etc., each a Tensor [B,C,Z,H,W] or [B,Z,H,W]
        projection: one of "max", "min", "mean"
        num_per_batch: how many samples to plot
    """
    images = batch["image_patch"]
    distlbls = batch["distance_patch"]
    lbls = batch["label_patch"]

    for img, distlbl, lbl in zip(images, distlbls, lbls):
        # Projections
        img_proj  = img[0].numpy().max(slice_dim)
        dist_proj = distlbl[0].numpy().min(slice_dim)
        lbl_proj  = lbl[0].numpy().max(slice_dim)

        # Plot side by side images
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, proj, title in zip(
            axes[:3],
            (img_proj, dist_proj, lbl_proj),
            ("Image Projection", "Distance Map", "Label Projection"),
        ):
            ax.imshow(proj)
            ax.set_title(title)
            ax.axis("off")
        
        # Compute stats
        stats = {
            'Image': (img_proj.min(), img_proj.max(), img_proj.mean()),
            'Distance': (dist_proj.min(), dist_proj.max(), dist_proj.mean()),
            'Label': (lbl_proj.min(), lbl_proj.max(), lbl_proj.mean()),
        }
        print(stats)
        # Bar chart of stats
        categories = list(stats.keys())
        mins = [stats[k][0] for k in categories]
        maxs = [stats[k][1] for k in categories]
        means = [stats[k][2] for k in categories]
        
        x = np.arange(len(categories))
        width = 0.2
        
        ax_stats = axes[3]
        ax_stats.bar(x - width, mins,    width, label='Min')
        ax_stats.bar(x,         means,  width, label='Mean')
        ax_stats.bar(x + width, maxs,    width, label='Max')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(categories)
        ax_stats.set_title('Min/Mean/Max per Projection')
        ax_stats.legend()
        ax_stats.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
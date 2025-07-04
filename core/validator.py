"""
Validator module for handling chunked inference in validation/test phases.
Now supports **both 2‑D (N, C, H, W)** and **3‑D (N, C, D, H, W)** inputs.
Implemented with robust size handling
and automatic dimensionality detection.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import process_in_chuncks  # unchanged – must support N‑D windows


class Validator:
    """Chunked, overlap‑tiled inference for 2‑D **or** 3‑D data.

    • Works with arbitrary batch size and channel count.
    • Pads the sample so every spatial dimension is divisible by a given
      *divisor* (default: 16) before tiling, then removes the pad.
    • Uses `patch_size` and `patch_margin` to create overlapping tiles.
      Only the *centre* region of each model prediction is kept and
      stitched together.

    Parameters
    ----------
    config : dict
        Dictionary with at least the keys:
            ``patch_size``   – tuple/list[int] (len == 2 or 3)
            ``patch_margin`` – tuple/list[int] same length as
                                 ``patch_size``
        Any other keys are ignored by this class.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patch_size: Tuple[int, ...] = tuple(config.get("patch_size", (512, 512)))
        self.patch_margin: Tuple[int, ...] = tuple(config.get("patch_margin", (32, 32)))
        self.logger = logging.getLogger(__name__)

        if len(self.patch_size) != len(self.patch_margin):
            raise ValueError(
                "patch_size %s and patch_margin %s must have the same number of dimensions"
                % (self.patch_size, self.patch_margin)
            )
        if len(self.patch_size) not in (2, 3):
            raise ValueError("Only 2‑D or 3‑D data are supported (got %d‑D)" % len(self.patch_size))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_div16_pad(size: int, divisor: int = 16) -> int:
        """Return how many voxels/pixels must be *added to the right* so that
        *size* becomes divisible by *divisor* (default 16)."""
        return (divisor - size % divisor) % divisor

    def _pad_to_valid_size(
        self, image: torch.Tensor, divisor: int = 16
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pad *image* so *all* spatial dims are divisible by ``divisor``.

        Only **right/bottom/back** padding is applied (no shift of origin).

        Parameters
        ----------
        image : torch.Tensor
            ``(N, C, H, W)`` or ``(N, C, D, H, W)`` tensor.
        divisor : int, optional
            The divisor (default 16).

        Returns
        -------
        image_padded : torch.Tensor
        pads         : list[int]
            Per‑dimension pad added (*same order as image spatial dims*).
        """

        spatial = image.shape[2:]
        pad_each: List[int] = [self._calc_div16_pad(s, divisor) for s in spatial]

        # Build pad tuple for F.pad – must be *reversed* order, one (left,right)
        # pair per dim starting with the last spatial dim.
        pad_tuple: List[int] = []
        for p in reversed(pad_each):
            pad_tuple.extend([0, p])  # (left = 0, right = p)

        if any(pad_each):
            try:
                image = F.pad(image, pad_tuple, mode="reflect")
            except RuntimeError:
                # "reflect" not implemented for 5‑D – fall back gracefully.
                image = F.pad(image, pad_tuple, mode="replicate")
        return image, pad_each

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    def run_chunked_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Full‑image/volume inference with overlapping tiles.

        Workflow (N‑D):
        1) **Margin pad** by ``patch_margin`` (reflect/replicate).
        2) **Div‑16 pad** so every spatial dim is divisible by 16.
        3) **Sliding‑window** inference:
            • window      = ``patch_size``
            • window step = ``patch_size − 2*patch_margin``
            • model is applied on each window; only the *centre* region
              is placed into the output canvas.
        4) **Remove** the div‑16 pad.
        5) **Remove** the initial margin pad.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(N, out_channels, *original_spatial*)``.
        """

        if device is None:
            device = next(model.parameters()).device

        model.eval()
        image = image.to(device)

        ndim = len(self.patch_size)  # 2 or 3
        if image.dim() != ndim + 2:
            raise ValueError(
                f"Input tensor dim {image.dim()} does not match patch_size ndim {ndim}"
            )

        # ----------------------------------------------------------
        # (A) First pad by the desired margins so borders get context
        # ----------------------------------------------------------
        if any(self.patch_margin):
            # Build pad tuple [ ... (left,right) per dim … ]
            pad_tuple: List[int] = []
            for m in reversed(self.patch_margin):
                pad_tuple.extend([m, m])
            try:
                image = F.pad(image, pad_tuple, mode="reflect")
            except RuntimeError:
                image = F.pad(image, pad_tuple, mode="replicate")

        # ----------------------------------------------------------
        # (B) Second pad until all dims divisible by 16
        # ----------------------------------------------------------
        padded_image, pad_div16 = self._pad_to_valid_size(image, 16)
        N, C, *spatial_pad = padded_image.shape

        # ----------------------------------------------------------
        # (C) Dummy forward to figure out #out channels
        # ----------------------------------------------------------
        with torch.no_grad():
            test_sizes = [
                min(s, p + 2 * m)
                for s, p, m in zip(spatial_pad, self.patch_size, self.patch_margin)
            ]
            slices: List[slice] = [slice(None), slice(None)] + [slice(0, t) for t in test_sizes]
            test_patch = padded_image[tuple(slices)]
            test_patch, _ = self._pad_to_valid_size(test_patch, 16)
            out_channels = model(test_patch).shape[1]

        # Allocate output canvas
        output_shape = (N, out_channels, *spatial_pad)
        output = torch.zeros(output_shape, device=device, dtype=padded_image.dtype)

        # ----------------------------------------------------------
        # (D) Sliding‑window inference
        # ----------------------------------------------------------
        def _process(chunk: torch.Tensor) -> torch.Tensor:  # noqa: D401
            with torch.no_grad():
                return model(chunk)

        with torch.no_grad():
            output = process_in_chuncks(
                padded_image,
                output,
                _process,
                list(self.patch_size),
                list(self.patch_margin),
            )

        # ----------------------------------------------------------
        # (E) Remove the div‑16 pad (right/bottom/back only)
        # ----------------------------------------------------------
        if any(pad_div16):
            slices: List[slice] = [slice(None), slice(None)]
            for p in pad_div16:
                slices.append(slice(None, -p if p else None))
            output = output[tuple(slices)]

        # ----------------------------------------------------------
        # (F) Remove the initial margin pad (all sides)
        # ----------------------------------------------------------
        if any(self.patch_margin):
            slices = [slice(None), slice(None)]
            for i, m in enumerate(self.patch_margin):
                end = output.shape[2 + i] - m
                slices.append(slice(m, end))
            output = output[tuple(slices)]

        return output

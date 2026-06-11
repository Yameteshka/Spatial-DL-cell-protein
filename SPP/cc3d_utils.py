"""
3D Connected Component Analysis Utilities for OME-Zarr Microglia Data

Shared utilities used by both the Napari test script and the ClearML
production script.

Key functions:
* ``scale_centroids_aniso``  -- anisotropic pixel-to-micrometre scaling
* ``label_connected_components_3d``  -- 3D CCL + regionprops + scaling
* ``process_zarr_masks``  -- load masks from a zarr root, run CCL on both
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


# Physical resolution constants
# Z-axis is much coarser than XY  (confocal microscopy)
RESOLUTION_UM = np.array([0.5, 0.11, 0.11], dtype=np.float64)  # (Z, Y, X)



# Step 0.2 Anisotropic scaling of centroid coordinates

def scale_centroids_aniso(
    centroids_px: np.ndarray,
    resolution: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Scale centroid coordinates from pixels to micrometres with anisotropic
    resolution.

    Parameters
    -
    centroids_px : np.ndarray
        Array of shape ``(N, 3)`` with columns ``(Z, Y, X)`` in **pixels**.
    resolution : np.ndarray | None
        Per-axis resolution ``(dZ, dY, dX)`` in um/px.
        Defaults to ``[0.5, 0.11, 0.11]``.

    Returns
    -
    np.ndarray
        Array of shape ``(N, 3)`` with columns ``(Z, Y, X)`` in **um**.
    """
    if resolution is None:
        resolution = RESOLUTION_UM
    resolution = np.asarray(resolution, dtype=np.float64)
    if centroids_px.ndim != 2 or centroids_px.shape[1] != 3:
        raise ValueError(
            f"centroids_px must be (N, 3), got {centroids_px.shape}"
        )
    return centroids_px.astype(np.float64) * resolution



# Step 0.3 3D connected component labelling + regionprops


@dataclass
class ObjectInfo:
    """Metadata for a single connected component."""
    label_id: int          # integer label in the labelled image
    centroid_z_um: float   # Z centroid in um
    centroid_y_um: float   # Y centroid in um
    centroid_x_um: float   # X centroid in um
    centroid_z_px: float   # Z centroid in px
    centroid_y_px: float   # Y centroid in px
    centroid_x_px: float   # X centroid in px
    volume_vox: int        # volume in voxels
    volume_um3: float      # volume in um^3  (anisotropic)

    # bounding box in pixels (z_min, y_min, x_min, z_max, y_max, x_max)
    bbox_zmin: int = 0
    bbox_ymin: int = 0
    bbox_xmin: int = 0
    bbox_zmax: int = 0
    bbox_ymax: int = 0
    bbox_xmax: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def label_connected_components_3d(
    binary_mask: np.ndarray,
    resolution: Optional[np.ndarray] = None,
    connectivity: int = 3,
) -> Dict[str, Any]:
    """
    Run 3D connected-component labelling on a binary mask, extract
    centroids and volumes via ``regionprops``, and scale centroids
    to physical coordinates (um).

    Parameters
    -
    binary_mask : np.ndarray
        3D binary mask of shape ``(Z, Y, X)``.
        Non-zero values are treated as foreground.
    resolution : np.ndarray | None
        Per-axis physical resolution ``(dZ, dY, dX)`` in um/px.
        Defaults to ``[0.5, 0.11, 0.11]``.
    connectivity : int
        3D connectivity for ``skimage.measure.label``.
        ``1`` = 6-neighbours, ``2`` = 18-neighbours, ``3`` = 26-neighbours.
        Default ``3`` (full 3D connectivity).

    Returns
    -
    dict
        ``"n_objects"``  : int   -- number of connected components
        ``"label_image"``: np.ndarray  -- integer-labelled image
        ``"objects"``     : list[ObjectInfo]  -- per-object metadata
        ``"centroids_um"``: np.ndarray  -- (N, 3) centroids in um (Z, Y, X)
        ``"centroids_px"``: np.ndarray  -- (N, 3) centroids in px (Z, Y, X)
        ``"volumes_vox"`` : np.ndarray  -- (N,) volumes in voxels
        ``"volumes_um3"`` : np.ndarray  -- (N,) anisotropic volumes in um^3
    """
    if resolution is None:
        resolution = RESOLUTION_UM
    resolution = np.asarray(resolution, dtype=np.float64)

    if binary_mask.ndim != 3:
        raise ValueError(
            f"binary_mask must be 3D (Z, Y, X), got {binary_mask.ndim}D"
        )

    #  Connected-component labelling 
    label_img = label(
        binary_mask.astype(bool),
        connectivity=connectivity,
    )
    n_objects = label_img.max()

    if n_objects == 0:
        logger.warning("No objects found in the binary mask.")
        empty_3 = np.empty((0, 3), dtype=np.float64)
        empty_1 = np.empty((0,), dtype=np.float64)
        return {
            "n_objects": 0,
            "label_image": label_img,
            "objects": [],
            "centroids_um": empty_3,
            "centroids_px": empty_3,
            "volumes_vox": empty_1.astype(int),
            "volumes_um3": empty_1,
        }

    #  Region properties 
    props = regionprops(label_img)

    centroids_px_list: List[np.ndarray] = []
    objects: List[ObjectInfo] = []

    # Anisotropic voxel volume: dZ * dY * dX  (um^3 per voxel)
    vox_vol_um3 = float(resolution[0] * resolution[1] * resolution[2])

    for p in props:
        # regionprops centroid is (Z, Y, X) for a 3D image
        cz_px, cy_px, cx_px = p.centroid
        vol_vox = int(p.area)  # "area" = number of voxels in 3D
        vol_um3 = vol_vox * vox_vol_um3

        centroids_px_list.append(np.array([cz_px, cy_px, cx_px]))

        # Bounding box: (min_row, min_col, ..., max_row, max_col, ...)
        # For 3D: (z_min, y_min, x_min, z_max, y_max, x_max)
        bb = p.bbox  # tuple of length 6 for 3D

        objects.append(ObjectInfo(
            label_id=int(p.label),
            centroid_z_um=0.0,  # filled after batch scaling
            centroid_y_um=0.0,
            centroid_x_um=0.0,
            centroid_z_px=float(cz_px),
            centroid_y_px=float(cy_px),
            centroid_x_px=float(cx_px),
            volume_vox=vol_vox,
            volume_um3=vol_um3,
            bbox_zmin=bb[0],
            bbox_ymin=bb[1],
            bbox_xmin=bb[2],
            bbox_zmax=bb[3],
            bbox_ymax=bb[4],
            bbox_xmax=bb[5],
        ))

    #  Batch anisotropic scaling of all centroids 
    centroids_px = np.stack(centroids_px_list)  # (N, 3)
    centroids_um = scale_centroids_aniso(centroids_px, resolution)

    # Fill in the um coordinates in each ObjectInfo
    for i, obj in enumerate(objects):
        obj.centroid_z_um = float(centroids_um[i, 0])
        obj.centroid_y_um = float(centroids_um[i, 1])
        obj.centroid_x_um = float(centroids_um[i, 2])

    volumes_vox = np.array([o.volume_vox for o in objects], dtype=int)
    volumes_um3 = np.array([o.volume_um3 for o in objects], dtype=np.float64)

    return {
        "n_objects": n_objects,
        "label_image": label_img,
        "objects": objects,
        "centroids_um": centroids_um,
        "centroids_px": centroids_px,
        "volumes_vox": volumes_vox,
        "volumes_um3": volumes_um3,
    }



# High-level: process both masks from a single zarr root


def process_zarr_masks(
    root,
    resolution: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load the two binary masks (cell_mask, protein_mask) from an OME-Zarr
    group and run 3D connected-component analysis on each.

    Parameters
    -
    root : zarr.Group
        The OME-Zarr root (or FOV sub-group) containing
        ``labels/cell_mask/0`` and ``labels/protein_mask/0``.
    resolution : np.ndarray | None
        Per-axis resolution ``(dZ, dY, dX)`` in um/px.

    Returns
    -
    dict
        ``"cell_mask"`` : result dict from ``label_connected_components_3d``
        ``"protein_mask"`` : result dict from ``label_connected_components_3d``
    """
    results: Dict[str, Dict[str, Any]] = {}

    mask_paths = {
        "cell_mask": "labels/cell_mask/0",
        "protein_mask": "labels/protein_mask/0",
    }

    for mask_name, zarr_path in mask_paths.items():
        try:
            # Navigate nested zarr groups: "labels" -> "cell_mask" -> "0"
            parts = zarr_path.split("/")
            arr = root
            for part in parts:
                arr = arr[part]
            mask_data = np.asarray(arr)  # (Z, Y, X) or (1, Z, Y, X)
            # Squeeze leading singleton dimension if present
            if mask_data.ndim == 4 and mask_data.shape[0] == 1:
                mask_data = mask_data[0]
            logger.info(
                "Loaded %s: shape=%s, dtype=%s, nonzero=%d",
                mask_name, mask_data.shape, mask_data.dtype,
                np.count_nonzero(mask_data),
            )
        except (KeyError, IndexError) as exc:
            logger.warning(
                "Could not load mask '%s' from zarr path '%s': %s",
                mask_name, zarr_path, exc,
            )
            continue

        logger.info("Running 3D CCL on %s ...", mask_name)
        result = label_connected_components_3d(
            mask_data, resolution=resolution, connectivity=3,
        )
        logger.info(
            "%s: found %d connected components",
            mask_name, result["n_objects"],
        )
        results[mask_name] = result

    return results

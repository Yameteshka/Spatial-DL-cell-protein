#!/usr/bin/env python3
"""
Production 3D Connected Component Analysis  --  ClearML Pipeline
=================================================================

Runs 3D connected-component labelling on ALL binary masks across the
entire cohort (Putamen + SN, HC + PD), and uploads results to MinIO.

MinIO data layout
-----------------
    YOUR_BASE_PREFIX/{HC|PD}/{region}/{patient}/
        Patient_ImageStack_0000.ome.zarr/
        Patient_ImageStack_0001.ome.zarr/
        ...

Windows compatibility
---------------------
On Windows the fsspecIO daemon thread crashes with socket errors
(WinError 10038) when zarr reads directly from an S3-backed store.
This script downloads each zarr store to a local temp directory before
opening it, which avoids the async I/O issues entirely.

ClearML integration
-------------------
* Project : SPP_training
* Queue   : default
* Results : uploaded to ``YOUR_STORAGE_NAME/spp-results/`` on MinIO

Output artifacts (per patient/FOV)
----------------------------------
For each processed FOV, a JSON file is written containing:

  Identifiers:
    patient_id, group, region, fov_id

  FOV spatial window (for SPP edge correction):
    fov_window_um: {z_min_um, z_max_um, y_min_um, y_max_um, x_min_um, x_max_um,
                    shape_zyx, resolution_zyx}
    resolution_zyx_um: [dZ, dY, dX] in um/px

  Point pattern data (centroids = SPP points, volume = mark):
    cell_mask:   list of ObjectInfo dicts (full detail per object)
                 Each: {label_id, centroid_z/y/x_um, centroid_z/y/x_px,
                        volume_um3, volume_vox, bbox_zmin...bbox_xmax}
    protein_mask: list of ObjectInfo dicts (same structure)
    n_cells, n_proteins: object counts

  SPP-ready structure (spp_data key):
    spp_data.type_I_cell.points_um:     [[z, y, x], ...] in um  (cell centroids)
    spp_data.type_I_cell.marks:         {volume_um3: [...], volume_vox: [...]}
    spp_data.type_II_protein.points_um: [[z, y, x], ...] in um  (protein centroids)
    spp_data.type_II_protein.marks:     {volume_um3: [...], volume_vox: [...]}
    spp_data.*.n_points:                int

  For SPP: cell centroids = Type I points, protein centroids = Type II points,
           volume_um3 = mark for marked point pattern analysis.

A single **summary CSV** aggregating all FOVs is also produced.

Usage
-----
    # Run locally (for debugging):
    CLEARML_DISABLED=true python cc3d_production.py

    # The script auto-submits to ClearML queue "default" on first run.
    python cc3d_production.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Windows asyncio / SSL fix  --  MUST be before s3fs/aiohttp import
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np

# ---------------------------------------------------------------------------
# MinIO env vars (same as zarr_patch_dataset.py)
# ---------------------------------------------------------------------------
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_MINIO_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_MINIO_SECRET_KEY"
os.environ["AWS_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["CLEARML_AGENT_BOTO3_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

import s3fs
import zarr
from skimage.measure import label, regionprops

# ===================================================================
# INLINE: cc3d_utils  (self-contained — no external module dependency)
# ===================================================================

# Physical resolution constants  (confocal microscopy, Z much coarser)
RESOLUTION_UM = np.array([0.5, 0.11, 0.11], dtype=np.float64)  # (Z, Y, X)


def scale_centroids_aniso(
    centroids_px: np.ndarray,
    resolution: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scale centroids from pixels to micrometres (anisotropic)."""
    if resolution is None:
        resolution = RESOLUTION_UM
    resolution = np.asarray(resolution, dtype=np.float64)
    if centroids_px.ndim != 2 or centroids_px.shape[1] != 3:
        raise ValueError(f"centroids_px must be (N, 3), got {centroids_px.shape}")
    return centroids_px.astype(np.float64) * resolution


@dataclass
class ObjectInfo:
    """Metadata for a single connected component (centroids + marks)."""
    label_id: int
    centroid_z_um: float
    centroid_y_um: float
    centroid_x_um: float
    centroid_z_px: float
    centroid_y_px: float
    centroid_x_px: float
    volume_vox: int
    volume_um3: float
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
    """3D CCL + regionprops + anisotropic centroid/volume extraction."""
    if resolution is None:
        resolution = RESOLUTION_UM
    resolution = np.asarray(resolution, dtype=np.float64)

    if binary_mask.ndim != 3:
        raise ValueError(f"binary_mask must be 3D, got {binary_mask.ndim}D")

    label_img = label(binary_mask.astype(bool), connectivity=connectivity)
    n_objects = label_img.max()

    if n_objects == 0:
        empty_3 = np.empty((0, 3), dtype=np.float64)
        empty_1 = np.empty((0,), dtype=np.float64)
        return {
            "n_objects": 0, "label_image": label_img, "objects": [],
            "centroids_um": empty_3, "centroids_px": empty_3,
            "volumes_vox": empty_1.astype(int), "volumes_um3": empty_1,
        }

    props = regionprops(label_img)
    centroids_px_list = []
    objects = []
    vox_vol_um3 = float(resolution[0] * resolution[1] * resolution[2])

    for p in props:
        cz_px, cy_px, cx_px = p.centroid
        vol_vox = int(p.area)
        vol_um3 = vol_vox * vox_vol_um3
        centroids_px_list.append(np.array([cz_px, cy_px, cx_px]))
        bb = p.bbox
        objects.append(ObjectInfo(
            label_id=int(p.label),
            centroid_z_um=0.0, centroid_y_um=0.0, centroid_x_um=0.0,
            centroid_z_px=float(cz_px), centroid_y_px=float(cy_px), centroid_x_px=float(cx_px),
            volume_vox=vol_vox, volume_um3=vol_um3,
            bbox_zmin=bb[0], bbox_ymin=bb[1], bbox_xmin=bb[2],
            bbox_zmax=bb[3], bbox_ymax=bb[4], bbox_xmax=bb[5],
        ))

    centroids_px = np.stack(centroids_px_list)
    centroids_um = scale_centroids_aniso(centroids_px, resolution)
    for i, obj in enumerate(objects):
        obj.centroid_z_um = float(centroids_um[i, 0])
        obj.centroid_y_um = float(centroids_um[i, 1])
        obj.centroid_x_um = float(centroids_um[i, 2])

    volumes_vox = np.array([o.volume_vox for o in objects], dtype=int)
    volumes_um3 = np.array([o.volume_um3 for o in objects], dtype=np.float64)
    return {
        "n_objects": n_objects, "label_image": label_img, "objects": objects,
        "centroids_um": centroids_um, "centroids_px": centroids_px,
        "volumes_vox": volumes_vox, "volumes_um3": volumes_um3,
    }


def process_zarr_masks(
    root,
    resolution: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load cell_mask & protein_mask from zarr, run 3D CCL on each."""
    results = {}
    mask_paths = {
        "cell_mask": "labels/cell_mask/0",
        "protein_mask": "labels/protein_mask/0",
    }
    for mask_name, zarr_path in mask_paths.items():
        try:
            parts = zarr_path.split("/")
            arr = root
            for part in parts:
                arr = arr[part]
            mask_data = np.asarray(arr)
            if mask_data.ndim == 4 and mask_data.shape[0] == 1:
                mask_data = mask_data[0]
            logger.info("Loaded %s: shape=%s, nonzero=%d",
                        mask_name, mask_data.shape, np.count_nonzero(mask_data))
        except (KeyError, IndexError) as exc:
            logger.warning("Could not load '%s': %s", mask_name, exc)
            continue
        logger.info("Running 3D CCL on %s ...", mask_name)
        result = label_connected_components_3d(mask_data, resolution=resolution, connectivity=3)
        logger.info("%s: %d connected components", mask_name, result["n_objects"])
        results[mask_name] = result
    return results

# ===================================================================
# END INLINE cc3d_utils
# ===================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINIO_ENDPOINT = os.environ.get(
    "MINIO_ENDPOINT", "YOUR_MINIO_ENDPOINT"
)
MINIO_ACCESS_KEY = os.environ.get(
    "MINIO_ACCESS_KEY", "YOUR_MINIO_ACCESS_KEY"
)
MINIO_SECRET_KEY = os.environ.get(
    "MINIO_SECRET_KEY", "YOUR_MINIO_SECRET_KEY"
)
BUCKET = "YOUR_STORAGE_NAME"
BASE_FOLDER = "YOUR_BASE_PREFIX"
RESULTS_PREFIX = "spp-results/cc3d_analysis"

# Local temp directory for downloading zarr stores
LOCAL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "cc3d_zarr_cache")

# Cohort definition
REGIONS = ["putamen", "substantiaNigra"]
GROUPS = ["HC", "PD"]

# Known missing patients (from the spec)
MISSING_PATIENTS = {
    ("substantiaNigra", "PD13"),  # SN PD13 absent
    ("substantiaNigra", "HC5"),   # SN HC5 absent
}
# FOVs missing for specific patients (4-digit FOV IDs)
MISSING_FOVS = {
    ("substantiaNigra", "HC1", "0025"),
    ("substantiaNigra", "HC3", "0008"),
}

# Regex for FOV zarr filenames
_FOV_RE = re.compile(r"Patient_ImageStack_(\d+)\.ome\.zarr", re.IGNORECASE)


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _make_s3fs() -> s3fs.S3FileSystem:
    """Create an s3fs filesystem for MinIO."""
    return s3fs.S3FileSystem(
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
        client_kwargs={
            "endpoint_url": f"https://{MINIO_ENDPOINT}",
            "region_name": "us-east-1",
        },
        config_kwargs={
            "read_timeout": 900,
            "connect_timeout": 300,
            "retries": {"max_attempts": 10, "mode": "adaptive"},
        },
        asynchronous=False,
    )


def _retry_s3(fn, *args, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
    """Retry an S3 operation with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_transient = (
                "502" in err_str or "503" in err_str or "500" in err_str
                or "bad gateway" in err_str
                or "service unavailable" in err_str
                or "timeout" in err_str
                or "ssl" in err_str
            )
            if not is_transient or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + (0.5 * (attempt + 1))
            logger.warning(
                "S3 transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries, e, delay,
            )
            time.sleep(delay)
    raise last_exc


def _download_zarr_locally(fs: s3fs.S3FileSystem, zarr_key: str) -> str:
    """
    Download a zarr store from MinIO to a local temp directory.

    This avoids the fsspecIO daemon thread crash on Windows that occurs
    when zarr reads directly from an S3-backed store.

    Returns the local path to the downloaded zarr directory.
    """
    local_name = zarr_key.replace("/", "_")
    local_path = os.path.join(LOCAL_CACHE_DIR, local_name)

    # Check if already cached and valid
    if os.path.isdir(local_path):
        has_zarr_meta = (
            os.path.exists(os.path.join(local_path, ".zgroup"))
            or os.path.exists(os.path.join(local_path, ".zarray"))
            or os.path.exists(os.path.join(local_path, "0", ".zarray"))
        )
        if has_zarr_meta:
            logger.debug("Using cached zarr: %s", local_path)
            return local_path
        else:
            logger.warning("Corrupt cache at %s — re-downloading.", local_path)
            shutil.rmtree(local_path, ignore_errors=True)

    s3_path = f"{BUCKET}/{zarr_key}"
    logger.info("Downloading zarr: %s -> %s", s3_path, local_path)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    try:
        _retry_s3(fs.get, s3_path, local_path, recursive=True)
    except Exception:
        if os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        raise

    return local_path


def _is_zarr_store(fs: s3fs.S3FileSystem, s3_path: str) -> bool:
    """Check if s3_path is a valid zarr store."""
    return (
        fs.exists(f"{s3_path}/.zgroup")
        or fs.exists(f"{s3_path}/.zarray")
        or fs.exists(f"{s3_path}/0/.zarray")
    )


# ---------------------------------------------------------------------------
# Discovery: find all zarr stores across the cohort
# ---------------------------------------------------------------------------

def discover_all_stores(fs: s3fs.S3FileSystem) -> List[Dict[str, str]]:
    """
    Walk the MinIO bucket and discover all zarr stores for all
    patients, regions, and FOVs.

    Actual layout on MinIO:
        YOUR_BASE_PREFIX/{group}/{region}/{patient}/
            Patient_ImageStack_0000.ome.zarr/
            Patient_ImageStack_0001.ome.zarr/
            ...

    Returns a list of dicts, each with keys:
        zarr_key, group, region, patient_id, fov_id
    """
    prefix = BASE_FOLDER + "/"
    stores: List[Dict[str, str]] = []

    for grp in GROUPS:
        for reg in REGIONS:
            folder_prefix = f"{prefix}{grp}/{reg}/"
            bucket_prefix = f"{BUCKET}/{folder_prefix}"

            try:
                patient_entries = fs.ls(bucket_prefix, detail=False)
            except FileNotFoundError:
                logger.warning("Folder not found: %s", folder_prefix)
                continue

            for pe in patient_entries:
                rel = pe.removeprefix(BUCKET + "/").rstrip("/")
                entry_name = rel.replace(folder_prefix.rstrip("/"), "").strip("/")
                if not entry_name or entry_name.startswith("."):
                    continue

                # entry_name is the patient directory name (e.g. "HC1", "PD3")
                patient_id = entry_name

                # Skip known missing patients
                if (reg, patient_id) in MISSING_PATIENTS:
                    logger.info("Skipping missing patient: %s / %s", reg, patient_id)
                    continue

                s3_path = f"{BUCKET}/{rel}"

                # Check if this IS a zarr store itself (unlikely, but handle)
                if _is_zarr_store(fs, s3_path):
                    stores.append({
                        "zarr_key": rel,
                        "group": grp,
                        "region": reg,
                        "patient_id": patient_id,
                        "fov_id": "root",
                    })
                    continue

                # List Patient_ImageStack_XXXX.ome.zarr under the patient dir
                try:
                    fov_entries = fs.ls(s3_path, detail=False)
                except FileNotFoundError:
                    continue

                for fe in fov_entries:
                    fov_rel = fe.removeprefix(BUCKET + "/").rstrip("/")
                    fov_name = fov_rel.replace(rel + "/", "").strip("/")
                    if not fov_name or fov_name.startswith("."):
                        continue

                    # Extract FOV ID from filename
                    m = _FOV_RE.match(fov_name)
                    if not m:
                        continue

                    fov_id = m.group(1)  # e.g. "0000", "0001"

                    # Verify it's a valid zarr store
                    if not _is_zarr_store(fs, fe.rstrip("/")):
                        logger.debug("Not a zarr store, skipping: %s", fov_name)
                        continue

                    # Skip known missing FOVs
                    if (reg, patient_id, fov_id) in MISSING_FOVS:
                        logger.info(
                            "Skipping missing FOV: %s / %s / %s",
                            reg, patient_id, fov_id,
                        )
                        continue

                    stores.append({
                        "zarr_key": fov_rel,
                        "group": grp,
                        "region": reg,
                        "patient_id": patient_id,
                        "fov_id": fov_id,
                    })

    logger.info("Discovered %d zarr stores across all cohorts.", len(stores))
    return stores


# ---------------------------------------------------------------------------
# Process a single zarr store (FOV)
# ---------------------------------------------------------------------------

def process_single_fov(
    fs: s3fs.S3FileSystem,
    store_info: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Process a single FOV: download zarr locally, load masks, run CCL,
    return results dict.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
    store_info : dict
        Must contain: zarr_key, group, region, patient_id, fov_id

    Returns
    -------
    dict | None
        Results dict with keys: patient_id, group, region, fov_id,
        cell_mask, protein_mask, n_cells, n_proteins
    """
    zarr_key = store_info["zarr_key"]
    group = store_info["group"]
    region = store_info["region"]
    patient_id = store_info["patient_id"]
    fov_id = store_info["fov_id"]

    # Download zarr to local temp directory (avoids fsspecIO crash)
    try:
        local_path = _download_zarr_locally(fs, zarr_key)
    except Exception as exc:
        logger.error("Failed to download zarr %s: %s", zarr_key, exc)
        return None

    # Open the LOCAL zarr store (no async I/O issues)
    try:
        root = zarr.open(local_path, mode="r")
    except Exception as exc:
        logger.error("Failed to open local zarr %s: %s", local_path, exc)
        return None

    # Each Patient_ImageStack_XXXX.ome.zarr IS the FOV root directly
    fov_root = root

    # ------------------------------------------------------------------
    # Read mask shapes for FOV window bounds (needed for SPP edge correction)
    # ------------------------------------------------------------------
    mask_shape = None
    for mask_name_try in ["cell_mask", "protein_mask"]:
        try:
            parts = f"labels/{mask_name_try}/0".split("/")
            arr = fov_root
            for part in parts:
                arr = arr[part]
            mask_shape = np.asarray(arr).shape
            # Squeeze leading singleton dim if present
            if len(mask_shape) == 4 and mask_shape[0] == 1:
                mask_shape = mask_shape[1:]
            break
        except (KeyError, IndexError):
            continue

    # Compute FOV window bounds in micrometers (for edge correction in SPP)
    fov_window = None
    if mask_shape is not None and len(mask_shape) == 3:
        fov_window = {
            "z_min_um": 0.0,
            "z_max_um": float(mask_shape[0] * RESOLUTION_UM[0]),
            "y_min_um": 0.0,
            "y_max_um": float(mask_shape[1] * RESOLUTION_UM[1]),
            "x_min_um": 0.0,
            "x_max_um": float(mask_shape[2] * RESOLUTION_UM[2]),
            "shape_zyx": list(mask_shape),
            "resolution_zyx": list(RESOLUTION_UM),
        }

    # Run 3D CCL on both masks
    try:
        cc_results = process_zarr_masks(fov_root, resolution=RESOLUTION_UM)
    except Exception as exc:
        logger.error(
            "CCL failed for %s/%s/%s: %s", patient_id, region, fov_id, exc
        )
        return None

    # Build output dict — includes everything needed for SPP analysis
    result: Dict[str, Any] = {
        "patient_id": patient_id,
        "group": group,
        "region": region,
        "fov_id": fov_id,
        "zarr_key": zarr_key,
        "fov_window_um": fov_window,
        "resolution_zyx_um": list(RESOLUTION_UM),  # (dZ, dY, dX)
    }

    for mask_name in ["cell_mask", "protein_mask"]:
        if mask_name in cc_results:
            cc = cc_results[mask_name]
            result[mask_name] = [o.to_dict() for o in cc["objects"]]
            result[f"n_{mask_name.replace('_mask', '')}s"] = cc["n_objects"]
        else:
            result[mask_name] = []
            result[f"n_{mask_name.replace('_mask', '')}s"] = 0

    # ------------------------------------------------------------------
    # SPP-ready structure: explicit points + marks for Spatial Point
    # Process analysis.  This makes it trivial to load into a marked
    # point pattern object (e.g. spatstat-like Python equivalents).
    # ------------------------------------------------------------------
    spp_data: Dict[str, Any] = {}

    for mask_name, point_type in [("cell_mask", "type_I_cell"),
                                   ("protein_mask", "type_II_protein")]:
        objects = result.get(mask_name, [])
        if objects:
            # Points: (N, 3) array of centroids in micrometres (Z, Y, X)
            points_um = [
                [o["centroid_z_um"], o["centroid_y_um"], o["centroid_x_um"]]
                for o in objects
            ]
            # Marks: volume in um^3 (primary mark for marked SPP)
            marks_volume_um3 = [o["volume_um3"] for o in objects]
            # Secondary mark: volume in voxels
            marks_volume_vox = [o["volume_vox"] for o in objects]
        else:
            points_um = []
            marks_volume_um3 = []
            marks_volume_vox = []

        spp_data[point_type] = {
            "points_um": points_um,          # [[z, y, x], ...] in um
            "marks": {
                "volume_um3": marks_volume_um3,   # primary mark
                "volume_vox": marks_volume_vox,   # secondary mark
            },
            "n_points": len(points_um),
        }

    result["spp_data"] = spp_data

    # Free memory: delete the label_image arrays (they can be large)
    for mask_name in ["cell_mask", "protein_mask"]:
        if mask_name in cc_results:
            cc_results[mask_name].pop("label_image", None)

    # ------------------------------------------------------------------
    # Delete local zarr copy to free disk space (each zarr ~200-400 MB)
    # ------------------------------------------------------------------
    try:
        if os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
            logger.debug("Deleted local zarr: %s", local_path)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Upload results to MinIO
# ---------------------------------------------------------------------------

def _convert_numpy(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _convert_numpy(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def upload_results_to_minio(
    fs: s3fs.S3FileSystem,
    results: List[Dict[str, Any]],
    summary_csv_path: str,
) -> None:
    """
    Upload all per-FOV JSON results and the summary CSV to MinIO.

    MinIO layout:
        spp-results/cc3d_analysis/
            {group}/{region}/{patient_id}/{fov_id}.json
            summary.csv
    """
    for r in results:
        json_key = (
            f"{BUCKET}/{RESULTS_PREFIX}/"
            f"{r['group']}/{r['region']}/{r['patient_id']}/{r['fov_id']}.json"
        )
        # Convert numpy types to native Python before JSON serialization
        r_safe = _convert_numpy(r)
        json_bytes = json.dumps(r_safe, indent=2, ensure_ascii=False).encode("utf-8")

        try:
            with fs.open(json_key, "wb") as f:
                f.write(json_bytes)
        except Exception as exc:
            logger.error(
                "Failed to upload %s: %s", json_key, exc
            )

    # Upload summary CSV
    csv_key = f"{BUCKET}/{RESULTS_PREFIX}/summary.csv"
    try:
        _retry_s3(fs.put, summary_csv_path, csv_key, max_retries=3)
        logger.info("Uploaded summary CSV to %s", csv_key)
    except Exception as exc:
        logger.error("Failed to upload summary CSV: %s", exc)

    logger.info(
        "Uploaded %d per-FOV results + summary CSV to MinIO.",
        len(results),
    )


# ---------------------------------------------------------------------------
# Write summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    results: List[Dict[str, Any]],
    csv_path: str,
) -> None:
    """
    Write a flat summary CSV with one row per FOV.

    Columns: patient_id, group, region, fov_id, n_cells, n_proteins,
             cell_vol_min_um3, cell_vol_max_um3, cell_vol_median_um3,
             cell_vol_mean_um3, protein_vol_min_um3, protein_vol_max_um3,
             protein_vol_median_um3, protein_vol_mean_um3
    """
    rows = []
    for r in results:
        row = {
            "patient_id": r["patient_id"],
            "group": r["group"],
            "region": r["region"],
            "fov_id": r["fov_id"],
            "n_cells": r.get("n_cells", 0),
            "n_proteins": r.get("n_proteins", 0),
        }

        for mask_name, prefix in [("cell_mask", "cell"), ("protein_mask", "protein")]:
            vols = [o["volume_um3"] for o in r.get(mask_name, [])]
            if vols:
                row[f"{prefix}_vol_min_um3"] = min(vols)
                row[f"{prefix}_vol_max_um3"] = max(vols)
                row[f"{prefix}_vol_median_um3"] = float(np.median(vols))
                row[f"{prefix}_vol_mean_um3"] = float(np.mean(vols))
            else:
                row[f"{prefix}_vol_min_um3"] = 0.0
                row[f"{prefix}_vol_max_um3"] = 0.0
                row[f"{prefix}_vol_median_um3"] = 0.0
                row[f"{prefix}_vol_mean_um3"] = 0.0

        rows.append(row)

    if not rows:
        logger.warning("No results to write to CSV.")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV written to %s (%d rows)", csv_path, len(rows))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Run the full 3D CCL pipeline across the entire cohort."""
    t_start = time.time()

    fs = _make_s3fs()

    # ------------------------------------------------------------------
    # Step 1: Discover all zarr stores
    # ------------------------------------------------------------------
    logger.info("Step 1: Discovering all zarr stores ...")
    stores = discover_all_stores(fs)
    logger.info("Found %d FOV zarr stores to process.", len(stores))

    if not stores:
        logger.error("No stores found. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 2: Process each FOV
    # ------------------------------------------------------------------
    logger.info("Step 2: Running 3D CCL on all FOVs ...")
    results: List[Dict[str, Any]] = []
    failed: List[str] = []

    # Process sequentially to avoid overwhelming MinIO with connections.
    # If you want parallelism, increase max_workers, but be careful
    # with memory (each mask is ~35 MB in RAM) and with the local
    # download step (disk I/O contention).
    MAX_WORKERS = 2  # conservative; each FOV loads ~35 MB mask into RAM

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_store = {
            pool.submit(process_single_fov, fs, s): s for s in stores
        }
        done_count = 0
        for future in as_completed(future_to_store):
            store_info = future_to_store[future]
            done_count += 1
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed.append(store_info["zarr_key"])
            except Exception as exc:
                logger.error(
                    "Exception processing %s: %s", store_info["zarr_key"], exc
                )
                failed.append(store_info["zarr_key"])

            if done_count % 10 == 0 or done_count == len(stores):
                logger.info(
                    "  Processed %d/%d FOVs ...", done_count, len(stores)
                )
                print(
                    f"[CC3D] Processed {done_count}/{len(stores)} FOVs ...",
                    flush=True,
                )

    logger.info(
        "Step 2 complete: %d FOVs processed, %d failed.",
        len(results), len(failed),
    )
    if failed:
        logger.warning("Failed stores: %s", failed[:20])

    # ------------------------------------------------------------------
    # Step 3: Write summary CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(LOCAL_CACHE_DIR, "cc3d_summary.csv")
    logger.info("Step 3: Writing summary CSV ...")
    write_summary_csv(results, csv_path)

    # ------------------------------------------------------------------
    # Step 4: Upload results to MinIO
    # ------------------------------------------------------------------
    logger.info("Step 4: Uploading results to MinIO ...")
    upload_results_to_minio(fs, results, csv_path)

    elapsed = time.time() - t_start
    logger.info(
        "Pipeline complete! Processed %d FOVs in %.1f seconds.",
        len(results), elapsed,
    )
    print(
        f"\n[DONE] Processed {len(results)} FOVs in {elapsed:.0f}s. "
        f"Results uploaded to MinIO at {RESULTS_PREFIX}/",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Aggregate statistics (for ClearML logging)
    # ------------------------------------------------------------------
    total_cells = sum(r.get("n_cells", 0) for r in results)
    total_proteins = sum(r.get("n_proteins", 0) for r in results)
    logger.info(
        "Aggregate: %d total cells, %d total protein aggregates across %d FOVs.",
        total_cells, total_proteins, len(results),
    )

    # Per-group statistics
    for grp in GROUPS:
        grp_results = [r for r in results if r["group"] == grp]
        grp_cells = sum(r.get("n_cells", 0) for r in grp_results)
        grp_proteins = sum(r.get("n_proteins", 0) for r in grp_results)
        logger.info(
            "  %s: %d cells, %d proteins across %d FOVs",
            grp, grp_cells, grp_proteins, len(grp_results),
        )

    # Per-region statistics
    for reg in REGIONS:
        reg_results = [r for r in results if r["region"] == reg]
        reg_cells = sum(r.get("n_cells", 0) for r in reg_results)
        reg_proteins = sum(r.get("n_proteins", 0) for r in reg_results)
        logger.info(
            "  %s: %d cells, %d proteins across %d FOVs",
            reg, reg_cells, reg_proteins, len(reg_results),
        )

    # Optional: clean up local cache to free disk space
    if os.path.isdir(LOCAL_CACHE_DIR):
        logger.info("Cleaning up local cache: %s", LOCAL_CACHE_DIR)
        shutil.rmtree(LOCAL_CACHE_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# ClearML entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check if ClearML is disabled (for local testing without ClearML)
    clearml_disabled = os.environ.get("CLEARML_DISABLED", "").lower() in (
        "1", "true", "yes",
    )

    if not clearml_disabled:
        from clearml import Task

        # NOTE: do NOT add scikit-image here — ClearML auto-detects it
        # from the `from skimage.measure import ...` import. Adding it
        # manually creates a duplicate requirement (scikit_image==X.Y.Z
        # + scikit-image) which crashes pip install.
        Task.add_requirements("boto3")
        Task.add_requirements("s3fs")

        task = Task.init(
            project_name="SPP_training",
            task_name="CC3D_Analysis_Pipeline",
            task_type=Task.TaskTypes.data_processing,
        )

        # This will submit the task to the remote queue and exit locally.
        # When the agent picks it up, it will re-run the script from here
        # and execute_remotely() will return (not exit), allowing the
        # rest of the script to run on the agent.
        task.execute_remotely(queue_name="default")

    # Run the pipeline (either locally or on a ClearML agent)
    run_pipeline()

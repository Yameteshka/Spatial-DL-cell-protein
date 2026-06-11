"""
Napari Interactive 3D Viewer  --  Connected Components Verification

Test script that downloads a **minimal** amount of data from MinIO
(one patient, one FOV), runs 3D connected-component labelling on both
binary masks (cell_mask + protein_mask), and opens an interactive
Napari viewer so you can visually verify that:

  1. Cells and protein aggregates are correctly detected.
  2. Centroids (shown as points) land at the centres of objects.
  3. Volumes make sense (visible object size vs. reported volume).

Configured object-store layout
    YOUR_BASE_PREFIX/{HC|PD}/{region}/{patient}/
        Patient_ImageStack_0000.ome.zarr/
        Patient_ImageStack_0001.ome.zarr/
        ...

Windows compatibility
On Windows the fsspecIO daemon thread crashes with socket errors
(WinError 10038) when zarr reads directly from an S3-backed store.
The workaround is to **download the zarr store to a local temp
directory first**, then open it locally with zarr.  This script does
that automatically.

Usage
    python cc3d_test_napari.py
    python cc3d_test_napari.py --patient PD3 --region substantiaNigra --fov 0000
    python cc3d_test_napari.py --local "C:/data/Patient_ImageStack_0000.ome.zarr"

Requirements
    pip install napari scikit-image numpy zarr s3fs
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
import time

# Apply the Windows asyncio/SSL workaround before importing s3fs or aiohttp.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np

# MinIO env vars (same as zarr_patch_dataset.py)
os.environ.setdefault("AWS_ACCESS_KEY_ID", "YOUR_MINIO_ACCESS_KEY")
os.environ.setdefault(
    "AWS_SECRET_ACCESS_KEY", "YOUR_MINIO_SECRET_KEY"
)
os.environ.setdefault(
    "AWS_ENDPOINT_URL",
    "YOUR_MINIO_ENDPOINT",
)
os.environ.setdefault(
    "CLEARML_AGENT_BOTO3_ENDPOINT_URL",
    "YOUR_MINIO_ENDPOINT",
)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import s3fs
import zarr

# Import our shared utilities (same directory)
from cc3d_utils import (
    RESOLUTION_UM,
    label_connected_components_3d,
    process_zarr_masks,
    scale_centroids_aniso,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Constants
MINIO_ENDPOINT = os.environ.get(
    "MINIO_ENDPOINT", "YOUR_MINIO_ENDPOINT"
)
MINIO_ACCESS_KEY = os.environ.get(
    "MINIO_ACCESS_KEY", "YOUR_MINIO_ACCESS_KEY"
)
MINIO_SECRET_KEY = os.environ.get(
    "MINIO_SECRET_KEY", "YOUR_MINIO_SECRET_KEY"
)
STORAGE_NAME = "YOUR_STORAGE_NAME"
BASE_PREFIX = "YOUR_BASE_PREFIX"

# Local temp directory for downloading zarr stores
CACHE_DIR = os.path.join(tempfile.gettempdir(), "YOUR_LOCAL_CACHE_DIR")

# Regex to extract FOV ID from the zarr filename:
#   Patient_ImageStack_0003.ome.zarr  ->  group(1) = "0003"
_FOV_RE = re.compile(r"Patient_ImageStack_(\d+)\.ome\.zarr", re.IGNORECASE)


# Helpers

def _make_s3fs():
    """Create an s3fs filesystem for MinIO (same config as zarr_patch_dataset)."""
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
    # Create a flat local path from the zarr key
    # e.g. "YOUR_BASE_PREFIX/HC/putamen/HC1/Patient_ImageStack_0000.ome.zarr"
    #   -> "YOUR_BASE_PREFIX_HC_putamen_HC1_Patient_ImageStack_0000.ome.zarr"
    local_name = zarr_key.replace("/", "_")
    local_path = os.path.join(CACHE_DIR, local_name)

    # Check if already cached and valid
    if os.path.isdir(local_path):
        has_zarr_meta = (
            os.path.exists(os.path.join(local_path, ".zgroup"))
            or os.path.exists(os.path.join(local_path, ".zarray"))
            or os.path.exists(os.path.join(local_path, "0", ".zarray"))
        )
        if has_zarr_meta:
            logger.info("Using cached zarr: %s", local_path)
            return local_path
        else:
            logger.warning("Corrupt cache at %s — re-downloading.", local_path)
            shutil.rmtree(local_path, ignore_errors=True)

    # Download from S3
    s3_path = f"{STORAGE_NAME}/{zarr_key}"
    logger.info("Downloading zarr from S3: %s -> %s", s3_path, local_path)
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        _retry_s3(fs.get, s3_path, local_path, recursive=True)
    except Exception:
        # Clean up on failure
        if os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        raise

    logger.info("Download complete: %s", local_path)
    return local_path


def _list_fov_zarrs(fs: s3fs.S3FileSystem, patient_prefix: str) -> list[dict]:
    """
    List all Patient_ImageStack_XXXX.ome.zarr stores under a patient
    directory on MinIO.

    Returns a list of dicts with keys: zarr_key, fov_id
    """
    fovs = []
    try:
        entries = fs.ls(patient_prefix, detail=False)
    except FileNotFoundError:
        return fovs

    for entry in entries:
        entry_name = entry.rstrip("/").split("/")[-1]
        m = _FOV_RE.match(entry_name)
        if m:
            fov_id = m.group(1)  # e.g. "0000"
            rel_key = entry.removeprefix(STORAGE_NAME + "/").rstrip("/")
            fovs.append({"zarr_key": rel_key, "fov_id": fov_id})

    # Sort by FOV ID
    fovs.sort(key=lambda x: x["fov_id"])
    return fovs


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Napari viewer for 3D connected components"
    )
    parser.add_argument(
        "--patient", default="HC1",
        help="Patient ID (default: HC1)",
    )
    parser.add_argument(
        "--group", default="HC",
        help="Cohort group: HC or PD (default: HC)",
    )
    parser.add_argument(
        "--region", default="putamen",
        help="Brain region: putamen or substantiaNigra (default: putamen)",
    )
    parser.add_argument(
        "--fov", default=None,
        help="FOV index, e.g. 0000 (default: auto-detect first FOV)",
    )
    parser.add_argument(
        "--local", default=None,
        help="Path to a local .zarr directory to open directly "
             "(skips MinIO download).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Delete any cached zarr data before downloading.",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Clean cache if requested
    if args.no_cache and os.path.isdir(CACHE_DIR):
        logger.info("Cleaning cache: %s", CACHE_DIR)
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    # Open zarr root
    if args.local:
        logger.info("Opening local zarr: %s", args.local)
        root = zarr.open(args.local, mode="r")
        fov_id = os.path.basename(args.local).split("_")[-1].replace(".ome.zarr", "")
    else:
        # Discover FOV zarr stores under the patient directory on MinIO.
        # Actual layout:
        #   YOUR_BASE_PREFIX/{group}/{region}/{patient}/Patient_ImageStack_XXXX.ome.zarr
        patient_prefix = (
            f"{STORAGE_NAME}/{BASE_PREFIX}/{args.group}/{args.region}/{args.patient}/"
        )
        logger.info("Listing FOV zarrs under: %s", patient_prefix)
        fs = _make_s3fs()
        fov_list = _list_fov_zarrs(fs, patient_prefix)

        if not fov_list:
            logger.error(
                "No FOV zarr stores found under %s. "
                "Check patient/group/region arguments.",
                patient_prefix,
            )
            sys.exit(1)

        logger.info("Found %d FOV zarr stores: %s",
                     len(fov_list),
                     [f["fov_id"] for f in fov_list])

        # Select the desired FOV
        if args.fov is not None:
            target_fov = args.fov.zfill(4)  # ensure 4-digit zero-padded
            selected = [f for f in fov_list if f["fov_id"] == target_fov]
            if not selected:
                logger.error(
                    "FOV '%s' not found. Available: %s",
                    target_fov, [f["fov_id"] for f in fov_list],
                )
                sys.exit(1)
            fov_info = selected[0]
        else:
            fov_info = fov_list[0]  # first FOV by default

        fov_id = fov_info["fov_id"]
        zarr_key = fov_info["zarr_key"]
        logger.info("Selected FOV %s: %s", fov_id, zarr_key)

        # Download the zarr store to local disk to avoid fsspecIO
        # daemon thread crash on Windows
        try:
            local_path = _download_zarr_locally(fs, zarr_key)
        except Exception as exc:
            logger.error("Failed to download zarr %s: %s", zarr_key, exc)
            sys.exit(1)

        # Open the LOCAL zarr store (no async I/O issues)
        logger.info("Opening local zarr: %s", local_path)
        try:
            root = zarr.open(local_path, mode="r")
        except Exception as exc:
            logger.error("Failed to open local zarr %s: %s", local_path, exc)
            sys.exit(1)

    logger.info("Using FOV: %s, root tree: %s", fov_id, dict(root) if root else "empty")

    # Load raw fluorescence + masks
    raw_channels = {}
    try:
        raw_arr = root["0"]
        raw_data = np.asarray(raw_arr)
        logger.info("Raw zarr array shape=%s, dtype=%s", raw_data.shape, raw_data.dtype)

        # Flexible channel detection:
        #   - (C, Z, Y, X) with C>=2: split by channel index
        #   - (Z, Y, X) no channel dim: treat as single channel
        #   - (1, Z, Y, X): squeeze leading dim
        if raw_data.ndim == 4:
            if raw_data.shape[0] >= 2:
                # Standard (C, Z, Y, X): C=0 -> pSyn, C=1 -> IBA1
                raw_channels["pSyn"] = raw_data[0].astype(np.float32)
                raw_channels["IBA1"] = raw_data[1].astype(np.float32)
            elif raw_data.shape[0] == 1:
                # (1, Z, Y, X) — squeeze leading dim
                raw_channels["IBA1"] = raw_data[0].astype(np.float32)
        elif raw_data.ndim == 3:
            # (Z, Y, X) — no channel dimension
            raw_channels["IBA1"] = raw_data.astype(np.float32)

        for ch_name, ch_data in raw_channels.items():
            logger.info(
                "Loaded raw %s: shape=%s, dtype=%s, range=[%.0f, %.0f]",
                ch_name, ch_data.shape, ch_data.dtype,
                ch_data.min(), ch_data.max(),
            )
    except (KeyError, IndexError) as exc:
        logger.warning("Could not load raw fluorescence: %s", exc)

    mask_data = {}
    for mask_name, zarr_path in [
        ("cell_mask", "labels/cell_mask/0"),
        ("protein_mask", "labels/protein_mask/0"),
    ]:
        try:
            parts = zarr_path.split("/")
            arr = root
            for part in parts:
                arr = arr[part]
            data = np.asarray(arr)
            if data.ndim == 4 and data.shape[0] == 1:
                data = data[0]
            mask_data[mask_name] = data
            logger.info(
                "Loaded %s: shape=%s, dtype=%s, nonzero=%d",
                mask_name, data.shape, data.dtype, np.count_nonzero(data),
            )
        except (KeyError, IndexError) as exc:
            logger.error("Failed to load %s: %s", mask_name, exc)

    if not mask_data:
        logger.error("No masks loaded. Exiting.")
        sys.exit(1)

    # Run connected-component analysis
    results = {}
    for mask_name, mask_arr in mask_data.items():
        logger.info("Running 3D CCL on %s ...", mask_name)
        result = label_connected_components_3d(
            mask_arr, resolution=RESOLUTION_UM, connectivity=3,
        )
        results[mask_name] = result
        logger.info(
            "%s: %d objects found. Volume range: [%d, %d] voxels",
            mask_name, result["n_objects"],
            int(result["volumes_vox"].min()) if len(result["volumes_vox"]) > 0 else 0,
            int(result["volumes_vox"].max()) if len(result["volumes_vox"]) > 0 else 0,
        )

    elapsed = time.time() - t0
    logger.info("Data loading + CCL completed in %.1f s", elapsed)

    logger.info("Connected-component results for patient %s, FOV %s:", args.patient, fov_id)
    for mask_name, result in results.items():
        logger.info("%s: %d objects", mask_name, result["n_objects"])
        if result["n_objects"] > 0:
            logger.info(
                "  Centroids (um): Z=[%.1f, %.1f] Y=[%.1f, %.1f] X=[%.1f, %.1f]",
                result['centroids_um'][:, 0].min(), result['centroids_um'][:, 0].max(),
                result['centroids_um'][:, 1].min(), result['centroids_um'][:, 1].max(),
                result['centroids_um'][:, 2].min(), result['centroids_um'][:, 2].max(),
            )
            logger.info(
                "  Volume (vox): min=%d, max=%d, median=%.0f",
                int(result['volumes_vox'].min()), int(result['volumes_vox'].max()),
                float(np.median(result['volumes_vox'])),
            )
            logger.info(
                "  Volume (um^3): min=%.2f, max=%.2f, median=%.2f",
                float(result['volumes_um3'].min()), float(result['volumes_um3'].max()),
                float(np.median(result['volumes_um3'])),
            )

    # Open Napari viewer — redesigned for clear 3D inspection
    # IMPORTANT: We add all layers in 2D mode FIRST, then switch to 3D.
    # This is a well-known workaround for napari/vispy issues where
    # Labels and Points layers fail to render when the viewer is
    # already in ndisplay=3 mode at the time of layer creation.
    logger.info("Opening Napari viewer ...")
    import napari

    # Detect napari version for API compatibility
    _napari_ver = tuple(
        int(x) for x in napari.__version__.split(".")[:2]
    )
    _use_border_api = _napari_ver >= (0, 5)
    logger.info("Napari version: %s (border_api=%s)", napari.__version__, _use_border_api)

    # Anisotropic scale: (dZ, dY, dX) in um/px
    scale = tuple(RESOLUTION_UM)  # (0.5, 0.11, 0.11)

    # Create the viewer in 2D mode so all layers load reliably.
    viewer = napari.Viewer(
        title=f"3D CC Verification — {args.patient} FOV {fov_id}",
        ndisplay=2,  # 2D first! Switch to 3D after all layers are added.
    )

    # Track how many layers are added successfully.
    _layer_count = 0

    # Layer 1: Raw fluorescence for tissue context.
    if "IBA1" in raw_channels:
        iba1 = raw_channels["IBA1"]
        if iba1.ndim == 3 and np.any(iba1 > 0):
            p1, p99 = np.percentile(iba1[iba1 > 0], [1, 99])
            try:
                viewer.add_image(
                    iba1.astype(np.float32),
                    name="IBA1 raw (green)",
                    scale=scale,
                    colormap="green",
                    contrast_limits=[p1, p99],
                    opacity=0.5,
                    blending="additive",
                )
                _layer_count += 1
                logger.info(f"  Loaded IBA1 raw data (green), shape={iba1.shape}")
            except Exception as exc:
                logger.warning(f"  Could not load IBA1 raw data: {exc}")

    if "pSyn" in raw_channels:
        psyn = raw_channels["pSyn"]
        if psyn.ndim == 3 and np.any(psyn > 0):
            p1, p99 = np.percentile(psyn[psyn > 0], [1, 99])
            try:
                viewer.add_image(
                    psyn.astype(np.float32),
                    name="pSyn raw (magenta)",
                    scale=scale,
                    colormap="magenta",
                    contrast_limits=[p1, p99],
                    opacity=0.35,
                    blending="additive",
                    visible=False,
                )
                _layer_count += 1
                logger.info(f"  Loaded pSyn raw data (magenta), shape={psyn.shape}")
            except Exception as exc:
                logger.warning(f"  Could not load pSyn raw data: {exc}")

    # Layer 2: Cell CC labels, used to verify detection.
    cell_result = results.get("cell_mask")
    if cell_result is not None and cell_result["n_objects"] > 0:
        cell_labels = cell_result["label_image"]

        try:
            viewer.add_labels(
                cell_labels,
                name=f"Cell CC labels ({cell_result['n_objects']} cells)",
                scale=scale,
                opacity=0.85,
            )
            _layer_count += 1
            logger.info(f"  Loaded cell labels: {cell_result['n_objects']} cells, shape={cell_labels.shape}")
        except Exception as exc:
            logger.warning(f"  Could not load cell labels: {exc}")

        # Cell centroids, shown as large red points with text labels.
        cell_centroids = cell_result["centroids_px"]  # (N, 3) in pixels

        pts_kw = {
            "name": "Cell centroids",
            "scale": scale,
            "size": 8.0,
            "face_color": "red",
            "symbol": "o",
        }
        if _use_border_api:
            pts_kw["border_color"] = "yellow"
            pts_kw["border_width"] = 1.0
            import pandas as pd
            pts_kw["features"] = pd.DataFrame({
                "label": [
                    f"Cell {i+1} ({cell_result['volumes_um3'][i]:.0f} um3)"
                    for i in range(len(cell_centroids))
                ],
            })
            pts_kw["text"] = {
                "string": "label",
                "size": 10,
                "color": "white",
                "anchor": "upper_left",
            }
        else:
            pts_kw["edge_color"] = "yellow"
            pts_kw["edge_width"] = 1.0
            pts_kw["text"] = {
                "string": np.array([
                    f"Cell {i+1} ({cell_result['volumes_um3'][i]:.0f} um3)"
                    for i in range(len(cell_centroids))
                ]),
                "size": 10,
                "color": "white",
                "anchor": "upper_left",
            }
        try:
            viewer.add_points(cell_centroids, **pts_kw)
            _layer_count += 1
            logger.info(f"  Loaded cell centroids: {len(cell_centroids)} points")
        except Exception as exc:
            logger.warning(f"  Could not load cell centroids: {exc}")

    # Layer 3: Protein centroids with VOLUME as mark (color + size)
    # Each protein centroid is colored and sized by its volume (µm³).
    # This makes the size distribution easier to inspect:
    #   - Small puncta (yellow)  → most common, diffuse Lewy puncta
    #   - Medium aggregates (orange/red) → moderate size
    #   - Large Lewy bodies (white/hot) → rare but biologically important
    #
    # In Napari, you can click any point to see its volume in the
    # features table. The colormap goes from dark purple to white:
    # green → yellow → red → white (plasma).

    prot_result = results.get("protein_mask")
    if prot_result is not None and prot_result["n_objects"] > 0:
        prot_centroids = prot_result["centroids_px"]  # (N, 3) in pixels
        prot_volumes_um3 = prot_result["volumes_um3"]  # (N,) in um3
        prot_volumes_vox = prot_result["volumes_vox"]  # (N,) in voxels

        # Compute size proportional to volume
        # Map volumes to point sizes: smallest=2, largest=12
        v_min, v_max = prot_volumes_um3.min(), prot_volumes_um3.max()
        if v_max > v_min:
            prot_sizes = 2.0 + 10.0 * (prot_volumes_um3 - v_min) / (v_max - v_min)
        else:
            prot_sizes = np.full(len(prot_volumes_um3), 4.0)

        # Build the features DataFrame with volume marks.
        import pandas as pd
        prot_features = pd.DataFrame({
            "volume_um3": prot_volumes_um3,
            "volume_vox": prot_volumes_vox,
            "log_volume": np.log10(prot_volumes_um3 + 1e-10),
        })

        # Protein centroids colored by volume, visible by default.
        prot_kw = {
            "name": "Protein (color=volume)",
            "scale": scale,
            "size": prot_sizes,
            "face_color": "log_volume",       # color mapped to log10(volume)
            "face_colormap": "plasma",         # purple(small) → yellow(large)
            "symbol": "o",
            "features": prot_features,
            "visible": True,                   # ON by default — this IS the key layer now
        }
        if _use_border_api:
            prot_kw["border_color"] = [0.8, 0.8, 0.8, 0.3]  # faint gray
            prot_kw["border_width"] = 0.2
        else:
            prot_kw["edge_color"] = [0.8, 0.8, 0.8, 0.3]
            prot_kw["edge_width"] = 0.2
        try:
            viewer.add_points(prot_centroids, **prot_kw)
            _layer_count += 1
            n_small = int(np.sum(prot_volumes_vox <= 20))
            n_med = int(np.sum((prot_volumes_vox > 20) & (prot_volumes_vox <= 50)))
            n_large = int(np.sum(prot_volumes_vox > 50))
            logger.info(f"  Loaded protein points (color=volume): {len(prot_centroids)} points")
            logger.info(f"       Size distribution: ≤20 vox={n_small}  21-50 vox={n_med}  >50 vox={n_large}")
        except Exception as exc:
            logger.warning(f"  Could not load protein points (color=volume): {exc}")

        # Also add a uniform-size version for comparison (off by default).
        prot_kw2 = {
            "name": "Protein (uniform, all same)",
            "scale": scale,
            "size": 2.0,
            "face_color": "yellow",
            "symbol": "o",
            "features": prot_features,
            "visible": False,
        }
        if _use_border_api:
            prot_kw2["border_color"] = "orange"
            prot_kw2["border_width"] = 0.2
        else:
            prot_kw2["edge_color"] = "orange"
            prot_kw2["edge_width"] = 0.2
        try:
            viewer.add_points(prot_centroids, **prot_kw2)
            _layer_count += 1
            logger.info(f"  Loaded protein points (uniform): {len(prot_centroids)} points")
        except Exception as exc:
            logger.warning(f"  Could not load protein points (uniform): {exc}")

        # Protein CC label image (off by default; rendering can be slow).
        try:
            viewer.add_labels(
                prot_result["label_image"],
                name=f"Protein CC labels ({prot_result['n_objects']} objects)",
                scale=scale,
                opacity=0.5,
                visible=False,
            )
            _layer_count += 1
            logger.info(f"  Loaded protein CC labels: {prot_result['n_objects']} objects")
        except Exception as exc:
            logger.warning(f"  Could not load protein CC labels: {exc}")

        # Binary mask overlay (off by default).
        prot_mask = mask_data.get("protein_mask")
        if prot_mask is not None:
            try:
                viewer.add_labels(
                    prot_mask.astype(np.int32),
                    name="Protein binary mask",
                    scale=scale,
                    opacity=0.25,
                    visible=False,
                )
                _layer_count += 1
                logger.info(f"  Loaded protein binary mask, shape={prot_mask.shape}")
            except Exception as exc:
                logger.warning(f"  Could not load protein binary mask: {exc}")

    logger.info(f"\n  Added {_layer_count} layers in total.")

    # Switch to 3D mode after all layers have been added.
    # Napari handles the 2D to 3D transition more reliably than
    # adding layers directly in 3D mode.
    logger.info("  Switching to 3D mode...")
    try:
        viewer.dims.ndisplay = 3
        logger.info("  3D mode is active. Rotate with the left mouse button.")
    except Exception as exc:
        logger.warning(f"  Could not switch to 3D: {exc}")
        logger.info("  You can also switch manually by clicking the 3D button in the viewer.")

    # Camera and view settings.
    viewer.reset_view()

    # Print the usage help.
    cell_n = cell_result["n_objects"] if cell_result else 0
    prot_n = prot_result["n_objects"] if prot_result else 0

    logger.info("Napari viewer ready.")
    logger.info("Navigation: rotate with the left mouse button, zoom with the mouse wheel, and pan with the right mouse button or Shift+drag.")
    logger.info("Visible layers: IBA1 raw, cell labels (%d cells), cell centroids, protein points colored by volume, protein points with uniform color, pSyn raw, and the protein binary mask.", cell_n)
    logger.info("Protein points use the plasma colormap: dark purple for small puncta, green/cyan for medium aggregates, and yellow/orange for large Lewy bodies.")
    logger.info("Suggested workflow: rotate the scene, inspect aggregate proximity to cells, click a point to inspect volume_um3, and disable IBA1 raw to focus on protein patterns.")

    logger.info("Napari viewer opened. Close the window to exit.")
    napari.run()


if __name__ == "__main__":
    main()

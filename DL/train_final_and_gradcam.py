"""
Train Final Model on Full Dataset + Generate 3D Grad-CAM / AOR
1. Trains a single final model (A, B, C) on ALL patients (except outliers).
2. Selects 1 typical HC and 1 typical PD patient.
3. Generates Per-Slice 3D Grad-CAM and 3D AOR visualizations.
4. Uploads model weights and images to MinIO.
"""

from __future__ import annotations
import os

# MinIO env vars
os.environ["AWS_ACCESS_KEY_ID"]     = "YOUR_MINIO_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_MINIO_SECRET_KEY"
os.environ["AWS_ENDPOINT_URL"]      = "YOUR_MINIO_ENDPOINT"
os.environ["CLEARML_AGENT_BOTO3_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"

from clearml import Task
task = Task.init(project_name="YOUR_CLEARML_PROJECT", task_name="GradCAM_3D_Grad-CAM")
task.execute_remotely(queue_name="high_q_80")

# Imports
import io
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional

from zarr_patch_dataset import ZarrPatchDataset, patch_collate_fn, _make_s3fs
from intensity_normalization import run_full_pre_scan, IntensityNormalizer

logger = logging.getLogger(__name__)

# Config
MINIO_ENDPOINT  = "YOUR_MINIO_ENDPOINT"
MINIO_ACCESS    = "YOUR_MINIO_ACCESS_KEY"
MINIO_SECRET    = "YOUR_MINIO_SECRET_KEY"
MINIO_STORAGE_NAME    = "YOUR_STORAGE_NAME"
MINIO_OUTPUT_PREFIX = "YOUR_OUTPUT_PREFIX"
EXCLUDED_PATIENTS = {"PD8", "PD10", "HC9"}  # Outliers based on pre-scan stats and visual inspection.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Functions
def _smart_conv1_init(in_channels: int, pretrained_conv1_weight: torch.Tensor) -> nn.Conv2d:
    orig_weight = pretrained_conv1_weight.clone()
    avg_weight = orig_weight.mean(dim=1, keepdim=True)
    new_weight = avg_weight.repeat(1, in_channels, 1, 1)
    scale = in_channels / 3.0
    new_weight = new_weight / scale
    conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data = new_weight
    return conv1

def create_model(
        in_channels: int,
        dropout: float = 0.5,
        calibration_loader: Optional[DataLoader] = None,
        channel_slice: Optional[slice] = None,
) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = _smart_conv1_init(in_channels, model.conv1.weight.data)
    model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512, 2))

    for param in model.layer1.parameters(): param.requires_grad = False
    for param in model.layer2.parameters(): param.requires_grad = False

    if calibration_loader is not None:
        logger.info("Calibrating BN running stats on microscopy data ...")
        model = model.to(DEVICE)
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = 0.1
                module.training = True

        ch_slice = channel_slice if channel_slice is not None else (slice(0, 25) if in_channels == 25 else slice(None))

        with torch.no_grad():
            n_cal_batches = 0
            for tensors, metas in calibration_loader:
                tensors = tensors.to(DEVICE)
                B, C, Z, H, W = tensors.shape
                x = tensors.reshape(B, C * Z, H, W)
                x_input = x[:, ch_slice, :, :]
                _ = model(x_input)
                n_cal_batches += 1
                if n_cal_batches >= 50: break

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            module.track_running_stats = True

    return model

# MinIO Helper
def upload_bytes_to_minio(data: bytes, minio_key: str, fs) -> None:
    s3_uri = f"{MINIO_STORAGE_NAME}/{minio_key}"
    try:
        with fs.open(s3_uri, "wb") as f:
            f.write(data)
        logger.info("Uploaded -> s3://%s", s3_uri)
    except Exception as exc:
        logger.error("Failed to upload bytes -> %s: %s", s3_uri, exc)

# 3D Grad-CAM Implementation (Per-Slice)
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x_2d, class_idx=None):
        self.model.eval()
        output = self.model(x_2d)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=False)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(x_2d.shape[2], x_2d.shape[3]), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

# 3D Visualization (Matplotlib - Stable for Remote Servers)
def plot_gradcam_3d(raw_patch, cam_3d, patient_id, label, model_name, region):
    """
    Generates two PNG images in memory (bytes):
    1. Peak Z-slice + 3D AOR contour + Z-activation profile
    2. Per-slice Grad-CAM grid (Z-slices)
    """
    from scipy import ndimage

    group_name = "PD" if label == 1 else "HC"
    iba1 = raw_patch[0]  # (25, 256, 256)
    psyn = raw_patch[1]  # (25, 256, 256)
    Z = iba1.shape[0]

    # Compute 3D AOR.
    threshold_3d = np.percentile(cam_3d[cam_3d > 0], 80) if cam_3d.max() > 0 else 0.9
    aor_3d_mask = (cam_3d >= threshold_3d).astype(np.float32)
    aor_per_slice = aor_3d_mask.sum(axis=(1, 2))
    peak_z = int(np.argmax(aor_per_slice))

    # Figure 1: Peak Z-slice + 3D AOR contour.
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

    axes[0, 0].imshow(iba1[peak_z], cmap='Greens')
    axes[0, 0].set_title(f"IBA1 at Peak Z={peak_z}", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(psyn[peak_z], cmap='Reds')
    axes[0, 1].set_title(f"pSyn at Peak Z={peak_z}\n(Lewy bodies here?)", fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(psyn[peak_z], cmap='Greys_r')
    axes[0, 2].imshow(cam_3d[peak_z], cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[0, 2].set_title(f"Grad-CAM at Peak Z={peak_z}", fontsize=12, fontweight="bold")
    axes[0, 2].axis('off')

    aor_peak_2d = aor_3d_mask[peak_z]
    aor_dilated = ndimage.binary_dilation(aor_peak_2d > 0.5, iterations=2)
    contour = aor_dilated ^ (aor_peak_2d > 0.5)

    axes[1, 0].imshow(psyn[peak_z], cmap='Reds')
    contour_overlay = np.zeros((psyn.shape[1], psyn.shape[2], 4))
    contour_overlay[contour, 1] = 1.0  # Green
    contour_overlay[contour, 3] = 1.0  # Alpha
    axes[1, 0].imshow(contour_overlay)
    axes[1, 0].set_title(f"3D AOR contour on pSyn (Z={peak_z})", fontsize=12, fontweight="bold")
    axes[1, 0].axis('off')

    psyn_mip = psyn.max(axis=0)
    cam_mip = cam_3d.max(axis=0)
    axes[1, 1].imshow(psyn_mip, cmap='Greys_r')
    axes[1, 1].imshow(cam_mip, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("Classic MIP Grad-CAM (2D logic)", fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].plot(range(Z), cam_3d.reshape(Z, -1).mean(axis=1), color='blue', label='Mean CAM')
    axes[1, 2].plot(range(Z), aor_per_slice / max(aor_per_slice.max(), 1), color='red', label='AOR Area (norm)')
    axes[1, 2].axvline(x=peak_z, color='green', linestyle='--', label=f'Peak Z={peak_z}')
    axes[1, 2].set_title("Activation profile across Z-depth", fontsize=12)
    axes[1, 2].set_xlabel("Z-slice")
    axes[1, 2].legend()

    fig.suptitle(f"Model {model_name} | {group_name} Patient: {patient_id} | Region: {region}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", bbox_inches="tight")
    plt.close(fig)
    buf1.seek(0)

    # Figure 2: Per-slice Grad-CAM grid across Z.
    slice_indices = list(range(0, Z, max(1, Z // 8)))[:8]
    n_slices = len(slice_indices)

    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12), dpi=150)

    for col, z_idx in enumerate(slice_indices):
        axes[0, col].imshow(psyn[z_idx], cmap='Reds', vmin=0, vmax=psyn.max()+1e-8)
        axes[0, col].set_title(f"pSyn Z={z_idx}", fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(psyn[z_idx], cmap='Greys_r')
        axes[1, col].imshow(cam_3d[z_idx], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, col].set_title(f"Grad-CAM Z={z_idx}", fontsize=10)
        axes[1, col].axis('off')

        aor_z = aor_3d_mask[z_idx]
        aor_dil_z = ndimage.binary_dilation(aor_z > 0.5, iterations=1)
        contour_z = aor_dil_z ^ (aor_z > 0.5)

        axes[2, col].imshow(psyn[z_idx], cmap='Reds')
        contour_z_vis = np.zeros((psyn.shape[1], psyn.shape[2], 4))
        contour_z_vis[contour_z, 1] = 1.0
        contour_z_vis[contour_z, 3] = 1.0
        axes[2, col].imshow(contour_z_vis)
        axes[2, col].set_title(f"3D AOR Z={z_idx}", fontsize=10)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("pSyn\n(Lewy bodies)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Per-slice\nGrad-CAM", fontsize=12, fontweight="bold")
    axes[2, 0].set_ylabel("3D AOR\nContour", fontsize=12, fontweight="bold")

    fig.suptitle(f"Z-Slices Grad-CAM | {group_name} Patient: {patient_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", bbox_inches="tight")
    plt.close(fig)
    buf2.seek(0)

    return buf1.read(), buf2.read()

# Main Pipeline
def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Device: %s", DEVICE)

    region = "substantiaNigra"
    fs = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET)

    logger.info("Creating dataset...")
    dataset = ZarrPatchDataset(
        minio_endpoint=MINIO_ENDPOINT, minio_access_key=MINIO_ACCESS,
        minio_secret_key=MINIO_SECRET, storage_name=MINIO_STORAGE_NAME,
        base_prefix="YOUR_BASE_PREFIX", model_type="C",
        regions=[region], groups=["HC", "PD"],
        filter_empty=True, normalization="intensity_pipeline", augment=False,
    )

    dataset.patches = [p for p in dataset.patches if p.patient_id not in EXCLUDED_PATIENTS]
    logger.info("Patches after excluding outliers: %d", len(dataset.patches))

    dataset.preload_data(max_workers=8, max_ram_gb=64.0)

    logger.info("Running intensity pre-scan...")
    patient_stats, target_stats, _ = run_full_pre_scan(dataset, alpha=0.05, max_patches_per_patient=500)
    normalizer = IntensityNormalizer(patient_stats, target_stats)
    dataset._normalizer = normalizer

    hc_patients = sorted(set(p.patient_id for p in dataset.patches if p.label == 0))
    pd_patients = sorted(set(p.patient_id for p in dataset.patches if p.label == 1))
    target_hc = hc_patients[0]
    target_pd = pd_patients[0]
    logger.info("Selected for Grad-CAM: HC=%s, PD=%s", target_hc, target_pd)

    hc_patch_idx = [i for i, p in enumerate(dataset.patches) if p.patient_id == target_hc][0]
    pd_patch_idx = [i for i, p in enumerate(dataset.patches) if p.patient_id == target_pd][0]

    logger.info("Training final model on 100%% data...")
    final_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=patch_collate_fn)

    models = {
        "A": create_model(25, 0.5, final_loader, slice(0, 25)).to(DEVICE),
        "B": create_model(25, 0.5, final_loader, slice(25, 50)).to(DEVICE),
        "C": create_model(50, 0.5, final_loader, slice(None)).to(DEVICE),
    }

    optimizer = AdamW([p for m in models.values() for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    dataset.augment = True
    for epoch in range(1, 16):
        for m in models.values():
            m.train()
            for module in m.modules():
                if isinstance(module, nn.BatchNorm2d): module.eval()

        epoch_loss = 0.0
        n_batches = 0

        for tensors, metas in final_loader:
            tensors = tensors.to(DEVICE)
            labels = torch.tensor(metas["label"], dtype=torch.long, device=DEVICE)
            B, C, Z, H, W = tensors.shape
            x = tensors.reshape(B, C*Z, H, W)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=DEVICE)
            inputs = {"A": x[:, 0:25], "B": x[:, 25:50], "C": x}

            for name, model in models.items():
                logits = model(inputs[name])
                total_loss += criterion(logits, labels)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/15 | Loss: %.4f", epoch, avg_loss)

    dataset.augment = False

    logger.info("Uploading final model weights to MinIO...")
    for name, model in models.items():
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        ckpt_key = f"{MINIO_OUTPUT_PREFIX}/{name}_cnn/{region}/checkpoints/final_model_{name}.pth"
        upload_bytes_to_minio(buf.read(), ckpt_key, fs)

    # 3D Grad-CAM Generation
    logger.info("Generating 3D Grad-CAM visualizations...")

    for label_type, patch_idx in [("HC", hc_patch_idx), ("PD", pd_patch_idx)]:
        raw_tensor, meta = dataset[patch_idx]
        patient_id = meta["patient_id"]
        label = meta["label"]

        input_tensor = raw_tensor.unsqueeze(0).to(DEVICE)
        B, C, Z, H, W = input_tensor.shape
        raw_numpy = raw_tensor.cpu().numpy()

        for name, model in models.items():
            logger.info("Processing Model %s for %s patient %s", name, label_type, patient_id)
            grad_cam = GradCAM3D(model, model.layer4[-1])

            in_channels = 25 if name in ["A", "B"] else 50
            cam_3d = np.zeros((Z, H, W), dtype=np.float32)
            zero_template = torch.zeros(B, in_channels, H, W, device=DEVICE)

            # Per-slice 3D Grad-CAM loop
            for z_idx in range(Z):
                x_2d = zero_template.clone()

                if name == "A":
                    # Model A: IBA1 only.
                    x_2d[0, z_idx, :, :] = input_tensor[0, 0, z_idx, :, :]
                elif name == "B":
                    # Model B: pSyn only.
                    x_2d[0, z_idx, :, :] = input_tensor[0, 1, z_idx, :, :]
                else:
                    # Model C: both channels.
                    x_2d[0, z_idx, :, :] = input_tensor[0, 0, z_idx, :, :]
                    x_2d[0, 25 + z_idx, :, :] = input_tensor[0, 1, z_idx, :, :]

                cam_3d[z_idx] = grad_cam(x_2d, class_idx=label)

            # Collect the image bytes.
            aor_png_bytes, perslice_png_bytes = plot_gradcam_3d(
                raw_patch=raw_numpy,
                cam_3d=cam_3d,
                patient_id=patient_id,
                label=label,
                model_name=name,
                region=region
            )

            # Save to MinIO using the new path layout.
            # The previous files were stored under: YOUR_OUTPUT_PREFIX/C_cnn/region/gradcam/...
            # The new files go to: YOUR_OUTPUT_PREFIX/C_cnn/region/gradcam_3D_Grad-CAM/...

            aor_key = f"{MINIO_OUTPUT_PREFIX}/{name}_cnn/{region}/gradcam_3D_Grad-CAM/GradCAM3D_{name}_{label_type}_{patient_id}_AOR.png"
            upload_bytes_to_minio(aor_png_bytes, aor_key, fs)

            perslice_key = f"{MINIO_OUTPUT_PREFIX}/{name}_cnn/{region}/gradcam_3D_Grad-CAM/GradCAM3D_{name}_{label_type}_{patient_id}_PerSlice.png"
            upload_bytes_to_minio(perslice_png_bytes, perslice_key, fs)

    logger.info("Pipeline complete!")

if __name__ == "__main__":
    main()

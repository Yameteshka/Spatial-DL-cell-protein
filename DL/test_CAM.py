"""
Local 3D Grad-CAM analysis in native 3D space
Architecture carried over from train_unified_5fold.py (Smart Conv1 Init, BN Calibration)
Generates per-slice Grad-CAM and computes AOR in the 3D volume
"""
import os
import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "gradcam_results_3d"

# Architecture reused from train_unified_5fold.py
def _smart_conv1_init(in_channels: int, pretrained_conv1_weight: torch.Tensor) -> nn.Conv2d:
    """Smart initialization from ImageNet weights used in the training pipeline."""
    orig_weight = pretrained_conv1_weight.clone()
    avg_weight = orig_weight.mean(dim=1, keepdim=True)
    new_weight = avg_weight.repeat(1, in_channels, 1, 1)
    scale = in_channels / 3.0
    new_weight = new_weight / scale

    conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data = new_weight
    return conv1

def load_model_for_inference(ckpt_path, in_channels=50):
    """Load the model exactly as during training (Smart Init + Freeze)."""
    # Load the pretrained weights first so conv1 can be initialized correctly.
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Replace conv1 using the same Smart Init strategy as the training script.
    model.conv1 = _smart_conv1_init(in_channels, model.conv1.weight.data)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, 2))

    # Freeze layer1 and layer2.
    for param in model.layer1.parameters(): param.requires_grad = False
    for param in model.layer2.parameters(): param.requires_grad = False

    # Freeze BatchNorm, matching the post-calibration training setup.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    logger.info("Loading checkpoint from %s", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)

    ckpt_in_channels = state_dict['conv1.weight'].shape[1]
    if ckpt_in_channels != in_channels:
        logger.warning("Checkpoint has %d channels, adjusting model.", ckpt_in_channels)
        model.conv1 = _smart_conv1_init(ckpt_in_channels, model.conv1.weight.data)
        in_channels = ckpt_in_channels

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model, in_channels

# 3D Grad-CAM (Per-slice)
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
        """Return a 2D activation map for the given slice."""
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


# Read and normalize the patch.
def read_patch_from_zarr(zarr_path, y_start=400, x_start=400, patch_size=256):
    root = zarr.open(zarr_path, mode='r')
    raw = root['0']
    Z = raw.shape[1]
    target_z = min(Z, 25)

    patch = np.asarray(raw[1, :target_z, y_start:y_start+patch_size, x_start:x_start+patch_size]).astype(np.float32)
    patch_psyn = np.asarray(raw[0, :target_z, y_start:y_start+patch_size, x_start:x_start+patch_size]).astype(np.float32)

    if target_z < 25:
        pad = np.zeros((25 - target_z, patch_size, patch_size), dtype=np.float32)
        patch = np.concatenate([patch, pad], axis=0)
        patch_psyn = np.concatenate([patch_psyn, pad], axis=0)

    result = np.stack([patch, patch_psyn], axis=0)

    for c in range(2):
        ch_data = result[c]
        p1, p99 = np.percentile(ch_data, 1), np.percentile(ch_data, 99)
        if p99 - p1 > 1e-8:
            result[c] = np.clip(ch_data, p1, p99)
            result[c] = (result[c] - p1) / (p99 - p1)
        else:
            result[c] = 0.0
    return result

def read_multiple_patches(zarr_path, patch_size=256, fov_size=1200, stride=400):
    root = zarr.open(zarr_path, mode='r')
    raw = root['0']
    positions = []
    for y in range(0, fov_size - patch_size, stride):
        for x in range(0, fov_size - patch_size, stride):
            positions.append((y, x))

    patches = []
    for y, x in positions[:3]:  # Use 3 patches for speed.
        patch_iba1 = np.asarray(raw[1, :25, y:y + patch_size, x:x + patch_size]).astype(np.float32)
        patch_psyn = np.asarray(raw[0, :25, y:y + patch_size, x:x + patch_size]).astype(np.float32)
        result = np.stack([patch_iba1, patch_psyn], axis=0)
        for c in range(2):
            ch_data = result[c]
            p1, p99 = np.percentile(ch_data, 1), np.percentile(ch_data, 99)
            if p99 - p1 > 1e-8:
                result[c] = np.clip(ch_data, p1, p99)
                result[c] = (result[c] - p1) / (p99 - p1)
            else:
                result[c] = 0.0
        patches.append(((y, x), result))
    return patches

# 3D visualization
def plot_gradcam_3d(raw_patch, cam_3d, patient_id, label, save_name="GradCAM3D"):
    """
    Visualization:
    1. Find the Z-slice with the strongest activation (peak slice).
    2. Draw the 3D AOR contour on that slice.
    3. Show a grid of Z-slices (per-slice Grad-CAM).
    """
    group_name = "PD" if label == 1 else "HC"
    iba1 = raw_patch[0]  # (25, 256, 256)
    psyn = raw_patch[1]  # (25, 256, 256)
    Z = iba1.shape[0]

    # 1. Compute AOR in the 3D volume.
    threshold_3d = np.percentile(cam_3d[cam_3d > 0], 80)
    aor_3d_mask = (cam_3d >= threshold_3d).astype(np.float32)

    # Find the Z-slice where AOR is strongest.
    aor_per_slice = aor_3d_mask.sum(axis=(1, 2))
    peak_z = int(np.argmax(aor_per_slice))
    logger.info("3D AOR Peak activation at Z-slice: %d", peak_z)

    # Figure 1: Peak Z-slice + 3D AOR contour.
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

    # IBA1 at the peak depth.
    axes[0, 0].imshow(iba1[peak_z], cmap='Greens')
    axes[0, 0].set_title(f"IBA1 at Peak Z={peak_z}", fontsize=12)
    axes[0, 0].axis('off')

    # pSyn at the peak depth.
    axes[0, 1].imshow(psyn[peak_z], cmap='Reds')
    axes[0, 1].set_title(f"pSyn at Peak Z={peak_z}\n(Lewy bodies here?)", fontsize=12)
    axes[0, 1].axis('off')

    # Grad-CAM at the peak depth.
    axes[0, 2].imshow(psyn[peak_z], cmap='Greys_r')
    axes[0, 2].imshow(cam_3d[peak_z], cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[0, 2].set_title(f"Grad-CAM at Peak Z={peak_z}", fontsize=12, fontweight="bold")
    axes[0, 2].axis('off')

    # 3D AOR contour at the peak depth.
    from scipy import ndimage
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

    # Standard MIP for comparison.
    psyn_mip = psyn.max(axis=0)
    cam_mip = cam_3d.max(axis=0)
    axes[1, 1].imshow(psyn_mip, cmap='Greys_r')
    axes[1, 1].imshow(cam_mip, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("Classic MIP Grad-CAM (2D logic)", fontsize=12)
    axes[1, 1].axis('off')

    # AOR statistics across Z.
    axes[1, 2].plot(range(Z), cam_3d.reshape(Z, -1).mean(axis=1), color='blue', label='Mean CAM')
    axes[1, 2].plot(range(Z), aor_per_slice / max(aor_per_slice.max(), 1), color='red', label='AOR Area (norm)')
    axes[1, 2].axvline(x=peak_z, color='green', linestyle='--', label=f'Peak Z={peak_z}')
    axes[1, 2].set_title("Activation profile across Z-depth", fontsize=12)
    axes[1, 2].set_xlabel("Z-slice")
    axes[1, 2].legend()

    fig.suptitle(f"3D Grad-CAM | {group_name} Patient: {patient_id}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{save_name}_{group_name}_{patient_id}_3D_AOR.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 3D AOR: %s", save_path)

    # Figure 2: Per-slice Grad-CAM grid across Z.
    slice_indices = list(range(0, Z, max(1, Z // 8)))[:8]
    n_slices = len(slice_indices)

    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12), dpi=150)

    for col, z_idx in enumerate(slice_indices):
        # Row 1: pSyn on each slice.
        axes[0, col].imshow(psyn[z_idx], cmap='Reds', vmin=0, vmax=psyn.max()+1e-8)
        axes[0, col].set_title(f"pSyn Z={z_idx}", fontsize=10)
        axes[0, col].axis('off')

        # Row 2: Per-slice Grad-CAM
        axes[1, col].imshow(psyn[z_idx], cmap='Greys_r')
        axes[1, col].imshow(cam_3d[z_idx], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, col].set_title(f"Grad-CAM Z={z_idx}", fontsize=10)
        axes[1, col].axis('off')

        # Row 3: 3D AOR contour on each slice.
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

    save_path = os.path.join(OUTPUT_DIR, f"{save_name}_{group_name}_{patient_id}_PerSlice.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Per-slice: %s", save_path)

# Main
def main():
    CKPT_PATH = "../models/best_model_C_fold_4.pth"

    ZARR_PATHS = {
        "HC": "data-dl/HC/substantiaNigra/HC1/Patient_ImageStack_0000.ome.zarr",
        "PD": "data-dl/PD/substantiaNigra/PD1/Patient_ImageStack_0000.ome.zarr",
    }

    model, in_channels = load_model_for_inference(CKPT_PATH, in_channels=50)
    grad_cam = GradCAM3D(model, model.layer4[-1])

    for label_type, zarr_path in ZARR_PATHS.items():
        if not os.path.exists(zarr_path):
            continue

        label = 0 if label_type == "HC" else 1
        logger.info("Reading patches from %s ...", zarr_path)

        patches = read_multiple_patches(zarr_path)

        for i, ((y, x), raw_patch) in enumerate(patches):
            input_tensor = torch.from_numpy(raw_patch).float().unsqueeze(0).to(DEVICE)
            B, C, Z, H, W = input_tensor.shape

            # Key change: per-slice Grad-CAM for the 50-channel model.
            cam_3d = np.zeros((Z, H, W), dtype=np.float32)

            # Create an empty tensor of shape (B, 50, H, W), matching training.
            zero_template = torch.zeros(B, C * Z, H, W, device=DEVICE)

            for z_idx in range(Z):
                # Copy the template so the original zero_template stays unchanged.
                x_2d = zero_template.clone()

                # Fill only the channels needed for the current Z-slice:
                # Channels 0-24 = IBA1 z-slices; write the current IBA1 slice to channel z_idx.
                x_2d[0, z_idx, :, :] = input_tensor[0, 0, z_idx, :, :]
                # Channels 25-49 = pSyn z-slices; write the current pSyn slice to channel 25 + z_idx.
                x_2d[0, 25 + z_idx, :, :] = input_tensor[0, 1, z_idx, :, :]

                # x_2d now has shape (1, 50, 256, 256), matching model C,
                # but only one Z-layer is populated.
                # This isolates activation at the selected depth.
                cam_3d[z_idx] = grad_cam(x_2d, class_idx=label)

            patient_id = os.path.basename(os.path.dirname(zarr_path))
            plot_gradcam_3d(
                raw_patch, cam_3d, f"{patient_id}_patch{i}_y{y}_x{x}",
                label,
                save_name=f"GradCAM3D_C_{label_type}",
            )

    logger.info("Done! Check the '%s' folder.", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
Visualize BEV perception on real CARLA frames.

Produces side-by-side images:
  Left: Front camera view
  Center: Predicted depth distribution
  Right: Bird's Eye View map (road + vehicle segmentation)

Usage:
    python scripts/visualize_bev.py
    python scripts/visualize_bev.py --checkpoint checkpoints/bev_model_best.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

from neurodriver.data.transforms import get_val_transforms
from neurodriver.models.bev_model import BEVDrivingModel
from neurodriver.utils.device import get_device


def visualize_frame(model, img_pil, device, transform):
    """Run BEV model on one frame and return visualization arrays."""
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Depth: expected depth per pixel
    depth_probs = output["depth_probs"][0].cpu()  # (D, h, w)
    depth_bins = model.bev.depth_bins.cpu()
    expected_depth = (depth_probs * depth_bins.view(-1, 1, 1)).sum(dim=0)  # (h, w)

    # BEV segmentation
    road_prob = torch.sigmoid(output["road_seg"][0, 0]).cpu().numpy()
    vehicle_prob = torch.sigmoid(output["vehicle_seg"][0, 0]).cpu().numpy()

    # Raw BEV features for heatmap fallback
    bev_feats = output["bev_features"][0].cpu().numpy()  # (C, h, w)

    return {
        "depth_map": expected_depth.numpy(),
        "road_bev": road_prob,
        "vehicle_bev": vehicle_prob,
        "bev_features": bev_feats,
    }


def render_bev_composite(road_bev, vehicle_bev, bev_features=None):
    """
    Create a colored BEV image.

    If the model is untrained (segmentation is garbage), fall back to
    rendering the BEV feature activation magnitude as a heatmap —
    this still shows spatial structure from the pretrained ResNet.
    """
    h, w = road_bev.shape
    bev_rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Check if segmentation is meaningful (not all same value)
    road_range = road_bev.max() - road_bev.min()
    veh_range = vehicle_bev.max() - vehicle_bev.min()
    seg_is_useful = road_range > 0.2 or veh_range > 0.2

    if seg_is_useful:
        # Trained model: use segmentation
        bev_rgb[:] = [0.05, 0.05, 0.1]
        road_mask = road_bev > 0.3
        bev_rgb[road_mask] = [0.2, 0.2, 0.3]
        road_high = road_bev > 0.6
        bev_rgb[road_high] = [0.3, 0.3, 0.45]
        veh_mask = vehicle_bev > 0.3
        bev_rgb[veh_mask] = [0.9, 0.2, 0.2]
    elif bev_features is not None:
        # Untrained: show feature activation heatmap
        # This reveals spatial structure from pretrained backbone
        activation = np.abs(bev_features).mean(axis=0)  # (h, w)
        if activation.max() > 0:
            activation = activation / activation.max()
        # Apply colormap: low=dark blue, high=bright yellow
        bev_rgb[:, :, 0] = activation * 0.9  # R
        bev_rgb[:, :, 1] = activation * 0.8  # G
        bev_rgb[:, :, 2] = (1 - activation) * 0.4 + 0.1  # B
    else:
        # No features available, just show road/vehicle as-is
        bev_rgb[:] = [0.05, 0.05, 0.1]
        # Use continuous coloring instead of hard threshold
        bev_rgb[:, :, 0] = vehicle_bev * 0.7
        bev_rgb[:, :, 1] = road_bev * 0.5
        bev_rgb[:, :, 2] = 0.15

    # Ego vehicle indicator (center-bottom)
    ego_y = h - 3
    ego_x = w // 2
    for dy in range(-2, 3):
        for dx in range(-1, 2):
            y, x = ego_y + dy, ego_x + dx
            if 0 <= y < h and 0 <= x < w:
                bev_rgb[y, x] = [0.2, 0.9, 0.3]

    return np.clip(bev_rgb, 0, 1)


def plot_single_frame(img_pil, bev_output, save_path=None):
    """Plot one frame: camera | depth | BEV."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f0f23")

    # Camera image
    axes[0].imshow(np.array(img_pil))
    axes[0].set_title("Front Camera", color="white", fontsize=12)
    axes[0].axis("off")

    # Depth map — auto-scale to actual range for visibility
    depth = bev_output["depth_map"]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1.0:
        # Depth predictions are nearly uniform — show entropy instead
        # Higher entropy = more uncertain about depth
        depth_probs_np = bev_output.get("depth_probs_raw", None)
        if depth_probs_np is not None:
            entropy = -(depth_probs_np * np.log(depth_probs_np + 1e-8)).sum(axis=0)
            im = axes[1].imshow(entropy, cmap="inferno")
            axes[1].set_title("Depth Uncertainty", color="white", fontsize=12)
        else:
            im = axes[1].imshow(depth, cmap="plasma", vmin=d_min, vmax=d_max)
            axes[1].set_title(f"Depth ({d_min:.0f}-{d_max:.0f}m)", color="white", fontsize=12)
    else:
        im = axes[1].imshow(depth, cmap="plasma", vmin=d_min, vmax=d_max)
        axes[1].set_title("Predicted Depth (m)", color="white", fontsize=12)
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(labelcolor="white")

    # BEV map
    bev_img = render_bev_composite(bev_output["road_bev"], bev_output["vehicle_bev"],
                                   bev_output.get("bev_features"))
    # Flip so ego is at bottom, forward is up
    axes[2].imshow(bev_img, origin="lower",
                   extent=[*(-25, 25), *(0, 50)])
    axes[2].set_title("Bird's Eye View", color="white", fontsize=12)
    axes[2].set_xlabel("Lateral (m)", color="white", fontsize=9)
    axes[2].set_ylabel("Forward (m)", color="white", fontsize=9)
    axes[2].tick_params(colors="white", labelsize=7)
    axes[2].set_facecolor("#0f0f23")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.3, 0.3, 0.45), label="Road"),
        Patch(facecolor=(0.9, 0.2, 0.2), label="Vehicle"),
        Patch(facecolor=(0.2, 0.9, 0.3), label="Ego"),
    ]
    axes[2].legend(handles=legend_elements, loc="upper right",
                   fontsize=8, facecolor="#1a1a2e", edgecolor="white",
                   labelcolor="white")

    plt.suptitle("NeuroDriver — Lift-Splat-Shoot BEV Perception",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    plt.close()


def plot_sequence(model, frames, device, transform, save_path, n_frames=8):
    """Plot BEV for a sequence of frames."""
    step = max(1, len(frames) // n_frames)
    selected = frames[::step][:n_frames]

    fig, axes = plt.subplots(3, n_frames, figsize=(3 * n_frames, 9))
    fig.patch.set_facecolor("#0f0f23")

    for i, frame in enumerate(selected):
        img_pil = Image.open(frame["rgb_path"]).convert("RGB")
        bev_out = visualize_frame(model, img_pil, device, transform)

        # Row 0: Camera
        axes[0, i].imshow(np.array(img_pil))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Camera", color="white", fontsize=10)

        # Row 1: Depth
        axes[1, i].imshow(bev_out["depth_map"], cmap="plasma", vmin=2, vmax=50)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Depth", color="white", fontsize=10)

        # Row 2: BEV
        bev_img = render_bev_composite(bev_out["road_bev"], bev_out["vehicle_bev"],
                                       bev_out.get("bev_features"))
        axes[2, i].imshow(bev_img, origin="lower")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("BEV", color="white", fontsize=10)

        axes[0, i].set_title(f"t={i * step}", color="white", fontsize=8)

    plt.suptitle("NeuroDriver BEV — Camera to Bird's Eye View Over Time",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Sequence saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="BEV model checkpoint (None = untrained demo)")
    parser.add_argument("--data-root", default="data_raw/transfuser")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load or create model
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = BEVDrivingModel(pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded BEV model from {args.checkpoint}")
    else:
        print("No checkpoint — using untrained model for visualization demo")
        model = BEVDrivingModel(pretrained=True).to(device)

    model.eval()
    transform = get_val_transforms((256, 256))

    # Find frames
    root = Path(args.data_root)
    route_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not route_dirs:
        print(f"No routes found in {root}")
        return

    # Pick first route with frames
    frames = []
    for rd in route_dirs:
        rgb_dir = rd / "rgb"
        meas_dir = rd / "measurements"
        if not rgb_dir.exists():
            continue
        rgb_files = {}
        for ext in ("*.jpg", "*.png"):
            for f in rgb_dir.glob(ext):
                rgb_files[f.stem] = str(f)
        meas_files = {f.stem: str(f) for f in meas_dir.glob("*.json")}
        valid = sorted(rgb_files.keys() & meas_files.keys())
        if len(valid) > 10:
            for fid in valid:
                frames.append({"rgb_path": rgb_files[fid], "meas_path": meas_files[fid]})
            print(f"Using route: {rd.name} ({len(valid)} frames)")
            break

    if not frames:
        print("No valid frames found")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single frame visualization
    img_pil = Image.open(frames[0]["rgb_path"]).convert("RGB")
    bev_out = visualize_frame(model, img_pil, device, transform)
    plot_single_frame(img_pil, bev_out, str(out_dir / "bev_single_frame.png"))

    # Sequence visualization
    plot_sequence(model, frames[:100], device, transform,
                  str(out_dir / "bev_sequence.png"), n_frames=args.num_frames)

    print("\nDone! Check the output directory for visualizations.")


if __name__ == "__main__":
    main()
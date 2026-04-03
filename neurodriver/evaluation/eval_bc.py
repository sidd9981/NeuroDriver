"""
Evaluate the trained BC model by comparing its predictions against expert labels.

Generates:
  1. Console stats: MAE for steer/throttle/brake, speed prediction error
  2. A grid of validation images with predicted vs actual controls overlaid
  3. Scatter plots: predicted vs actual for each control signal

Usage:
    python scripts/eval_bc.py
    python scripts/eval_bc.py --checkpoint checkpoints/best.pt --num-samples 200
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurodriver.data.dataset import build_dataset
from neurodriver.data.transforms import denormalize
from neurodriver.models.e2e_model import DrivingModel
from neurodriver.utils.device import get_device


def load_model(checkpoint_path: str, device: torch.device) -> DrivingModel:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = DrivingModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False,  # Don't need ImageNet weights, we have our trained weights
        feature_dim=cfg["model"]["feature_dim"],
        speed_embed_dim=cfg["model"]["speed_embed_dim"],
        command_embed_dim=cfg["model"]["command_embed_dim"],
        num_commands=cfg["model"]["num_commands"],
        hidden_dims=cfg["model"]["policy"]["hidden_dims"],
        dropout=cfg["model"]["policy"]["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model, cfg


@torch.no_grad()
def collect_predictions(model, dataset, device, num_samples=200):
    """Run model on dataset and collect predictions vs ground truth."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_pred_steer = []
    all_pred_throttle = []
    all_pred_brake = []
    all_pred_speed = []
    all_gt_steer = []
    all_gt_throttle = []
    all_gt_brake = []
    all_gt_speed = []
    all_images = []
    all_commands = []

    count = 0
    for batch in loader:
        if count >= num_samples:
            break

        image = batch["image"].to(device)
        speed = batch["speed"].to(device)
        command = batch["command"].to(device)

        pred = model(image, speed, command)

        n = min(image.size(0), num_samples - count)

        all_pred_steer.append(pred["steer"][:n].cpu())
        all_pred_throttle.append(pred["throttle"][:n].cpu())
        all_pred_brake.append(pred["brake"][:n].cpu())
        all_pred_speed.append(pred["pred_speed"][:n].cpu())

        all_gt_steer.append(batch["steer"][:n])
        all_gt_throttle.append(batch["throttle"][:n])
        all_gt_brake.append(batch["brake"][:n])
        all_gt_speed.append(batch["speed"][:n])

        all_images.append(denormalize(batch["image"][:n]))
        all_commands.append(batch["command"][:n])

        count += n

    return {
        "pred_steer": torch.cat(all_pred_steer).squeeze(),
        "pred_throttle": torch.cat(all_pred_throttle).squeeze(),
        "pred_brake": torch.cat(all_pred_brake).squeeze(),
        "pred_speed": torch.cat(all_pred_speed).squeeze(),
        "gt_steer": torch.cat(all_gt_steer).squeeze(),
        "gt_throttle": torch.cat(all_gt_throttle).squeeze(),
        "gt_brake": torch.cat(all_gt_brake).squeeze(),
        "gt_speed": torch.cat(all_gt_speed).squeeze(),
        "images": torch.cat(all_images),
        "commands": torch.cat(all_commands).squeeze(),
    }


def print_stats(results):
    """Print evaluation statistics."""
    steer_mae = (results["pred_steer"] - results["gt_steer"]).abs().mean().item()
    throttle_mae = (results["pred_throttle"] - results["gt_throttle"]).abs().mean().item()
    brake_mae = (results["pred_brake"] - results["gt_brake"]).abs().mean().item()
    speed_mae = (results["pred_speed"] - results["gt_speed"]).abs().mean().item()

    steer_corr = np.corrcoef(
        results["pred_steer"].numpy(), results["gt_steer"].numpy()
    )[0, 1]
    throttle_corr = np.corrcoef(
        results["pred_throttle"].numpy(), results["gt_throttle"].numpy()
    )[0, 1]

    print("\n" + "=" * 50)
    print("  BC Model Evaluation Results")
    print("=" * 50)
    print(f"  Samples evaluated: {len(results['gt_steer'])}")
    print()
    print(f"  Steering    MAE: {steer_mae:.4f}   (correlation: {steer_corr:.3f})")
    print(f"  Throttle    MAE: {throttle_mae:.4f}   (correlation: {throttle_corr:.3f})")
    print(f"  Brake       MAE: {brake_mae:.4f}")
    print(f"  Speed pred  MAE: {speed_mae:.4f} m/s")
    print()

    # Distribution of expert commands
    cmds = results["commands"].numpy()
    cmd_names = {1: "Left", 2: "Right", 3: "Straight", 4: "Follow"}
    print("  Command distribution:")
    for c in sorted(cmd_names.keys()):
        pct = (cmds == c).mean() * 100
        print(f"    {cmd_names[c]:10s}: {pct:.1f}%")
    print("=" * 50)


def plot_scatter(results, save_path="checkpoints/bc_eval_scatter.png"):
    """Scatter plots: predicted vs actual for each control."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, name in zip(axes, ["steer", "throttle", "brake"]):
        pred = results[f"pred_{name}"].numpy()
        gt = results[f"gt_{name}"].numpy()

        ax.scatter(gt, pred, alpha=0.3, s=8, c="steelblue")

        # Perfect prediction line
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")

        ax.set_xlabel(f"Expert {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name.capitalize()}: MAE={abs(pred - gt).mean():.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nScatter plot saved to: {save_path}")
    plt.close()


def plot_driving_strip(results, save_path="checkpoints/bc_eval_driving.png"):
    """
    Show a strip of consecutive frames with predicted vs actual controls.
    Like watching the model 'drive' through a sequence.
    """
    n_frames = min(16, len(results["images"]))
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))

    for i in range(n_frames):
        row = i // 8
        col = i % 8
        ax = axes[row, col]

        # Show image
        img = results["images"][i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)

        # Overlay controls as text
        ps = results["pred_steer"][i].item()
        gs = results["gt_steer"][i].item()
        pt = results["pred_throttle"][i].item()
        gt_t = results["gt_throttle"][i].item()

        cmd_map = {1: "L", 2: "R", 3: "S", 4: "F"}
        cmd = cmd_map.get(results["commands"][i].item(), "?")

        ax.set_title(
            f"Cmd:{cmd}\n"
            f"St: {ps:+.3f} / {gs:+.3f}\n"
            f"Th: {pt:.2f} / {gt_t:.2f}",
            fontsize=7,
            color="lime" if abs(ps - gs) < 0.05 else "red",
        )
        ax.axis("off")

    plt.suptitle(
        "BC Model Predictions (green=good, red=off)  |  Format: predicted / expert",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Driving strip saved to: {save_path}")
    plt.close()


def plot_steering_timeseries(results, save_path="checkpoints/bc_eval_steering.png"):
    """Plot predicted vs actual steering over time."""
    n = min(200, len(results["gt_steer"]))

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    x = range(n)

    # Steering
    axes[0].plot(x, results["gt_steer"][:n].numpy(), "b-", label="Expert", alpha=0.7)
    axes[0].plot(x, results["pred_steer"][:n].numpy(), "r-", label="Predicted", alpha=0.7)
    axes[0].set_ylabel("Steering")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("BC Model: Predicted vs Expert Controls Over Time")

    # Throttle
    axes[1].plot(x, results["gt_throttle"][:n].numpy(), "b-", label="Expert", alpha=0.7)
    axes[1].plot(
        x, results["pred_throttle"][:n].numpy(), "r-", label="Predicted", alpha=0.7
    )
    axes[1].set_ylabel("Throttle")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Speed
    axes[2].plot(x, results["gt_speed"][:n].numpy(), "b-", label="Actual", alpha=0.7)
    axes[2].plot(
        x, results["pred_speed"][:n].numpy(), "r-", label="Predicted", alpha=0.7
    )
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Time series saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC model")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt", help="Model checkpoint"
    )
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Number of val samples to evaluate"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint, device)

    # Load validation data
    val_dataset = build_dataset(cfg, split="val")
    print(f"Validation set: {len(val_dataset)} frames")

    # Run evaluation
    print(f"\nRunning inference on {args.num_samples} samples...")
    results = collect_predictions(model, val_dataset, device, args.num_samples)

    # Print stats
    print_stats(results)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_scatter(results)
    plot_driving_strip(results)
    plot_steering_timeseries(results)

    print("\nDone! Check the checkpoints/ folder for plots.")


if __name__ == "__main__":
    main()
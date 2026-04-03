"""
Visualize BC model driving through a sequence of real CARLA frames.

Produces a video (MP4) or image strip showing:
  - Camera image from the ego vehicle
  - Steering wheel indicator (predicted vs expert)
  - Throttle/brake bars (predicted vs expert)
  - Speed readout
  - Navigation command

Usage:
    python scripts/visualize_bc_driving.py
    python scripts/visualize_bc_driving.py --route Town01_Rep0_route_000024 --num-frames 200
    python scripts/visualize_bc_driving.py --save-video  # saves MP4
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from neurodriver.data.transforms import get_val_transforms
from neurodriver.models.e2e_model import DrivingModel
from neurodriver.utils.device import get_device


CMD_NAMES = {1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "FOLLOW"}
CMD_COLORS = {1: "#ff6b6b", 2: "#ffd93d", 3: "#6bcb77", 4: "#4d96ff"}


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DrivingModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False,
        feature_dim=cfg["model"]["feature_dim"],
        speed_embed_dim=cfg["model"]["speed_embed_dim"],
        command_embed_dim=cfg["model"]["command_embed_dim"],
        num_commands=cfg["model"]["num_commands"],
        hidden_dims=cfg["model"]["policy"]["hidden_dims"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def load_route_frames(data_root, route_name, max_frames=200):
    """Load consecutive frames from a single route."""
    route_dir = Path(data_root) / route_name
    rgb_dir = route_dir / "rgb"
    meas_dir = route_dir / "measurements"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Route not found: {route_dir}")

    # Find all valid frames
    rgb_files = {}
    for ext in ("*.png", "*.jpg"):
        for f in rgb_dir.glob(ext):
            rgb_files[f.stem] = str(f)

    meas_files = {f.stem: str(f) for f in meas_dir.glob("*.json")}
    valid = sorted(rgb_files.keys() & meas_files.keys())[:max_frames]

    frames = []
    for fid in valid:
        with open(meas_files[fid]) as f:
            meas = json.load(f)
        frames.append({
            "rgb_path": rgb_files[fid],
            "meas": meas,
            "frame_id": fid,
        })
    return frames


def find_best_route(data_root):
    """Find a route with the most frames for visualization."""
    root = Path(data_root)
    best_route = None
    best_count = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        rgb_dir = d / "rgb"
        if rgb_dir.exists():
            count = len(list(rgb_dir.glob("*.jpg"))) + len(list(rgb_dir.glob("*.png")))
            if count > best_count:
                best_count = count
                best_route = d.name
    return best_route


def draw_steering_wheel(ax, pred_steer, gt_steer):
    """Draw a steering indicator showing predicted vs expert."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background arc
    theta = np.linspace(-np.pi * 0.8, np.pi * 0.8, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#333333", linewidth=8, alpha=0.3)

    # Expert steering (blue)
    gt_angle = -gt_steer * np.pi * 0.8
    ax.plot([0, 0.8 * np.cos(np.pi / 2 + gt_angle)],
            [0, 0.8 * np.sin(np.pi / 2 + gt_angle)],
            color="#4d96ff", linewidth=4, alpha=0.6, label="Expert")

    # Predicted steering (red)
    pred_angle = -pred_steer * np.pi * 0.8
    ax.plot([0, 0.9 * np.cos(np.pi / 2 + pred_angle)],
            [0, 0.9 * np.sin(np.pi / 2 + pred_angle)],
            color="#ff6b6b", linewidth=4, label="Predicted")

    # Center dot
    ax.plot(0, 0, "o", color="white", markersize=6)

    ax.set_title(f"Steer: {pred_steer:+.3f} (expert: {gt_steer:+.3f})",
                 fontsize=9, color="white")


def draw_control_bars(ax, pred_throttle, pred_brake, gt_throttle, gt_brake):
    """Draw throttle and brake bars."""
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.1)
    ax.axis("off")

    bar_width = 0.6

    # Throttle bars
    ax.barh(0.8, gt_throttle, height=bar_width * 0.4, color="#4d96ff",
            alpha=0.5, left=0)
    ax.barh(0.8, pred_throttle, height=bar_width * 0.4, color="#6bcb77",
            alpha=0.8, left=0)
    ax.text(-0.05, 0.8, "THR", fontsize=8, color="white", ha="right", va="center")
    ax.text(max(pred_throttle, gt_throttle) + 0.05, 0.8,
            f"{pred_throttle:.2f}/{gt_throttle:.2f}", fontsize=7,
            color="white", va="center")

    # Brake bars
    ax.barh(0.3, gt_brake, height=bar_width * 0.4, color="#4d96ff",
            alpha=0.5, left=0)
    ax.barh(0.3, pred_brake, height=bar_width * 0.4, color="#ff6b6b",
            alpha=0.8, left=0)
    ax.text(-0.05, 0.3, "BRK", fontsize=8, color="white", ha="right", va="center")
    ax.text(max(pred_brake, gt_brake) + 0.05, 0.3,
            f"{pred_brake:.2f}/{gt_brake:.2f}", fontsize=7,
            color="white", va="center")


@torch.no_grad()
def render_frame(model, frame, transform, device, fig, axes):
    """Render one frame with model predictions overlaid."""
    # Load and transform image
    img_pil = Image.open(frame["rgb_path"]).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    meas = frame["meas"]
    speed = torch.tensor([[meas.get("speed", 0.0)]], dtype=torch.float32, device=device)
    command = torch.tensor([[meas.get("command", 4)]], dtype=torch.long, device=device)

    # Run model
    pred = model(img_tensor, speed, command)

    pred_steer = pred["steer"].item()
    pred_throttle = pred["throttle"].item()
    pred_brake = pred["brake"].item()
    gt_steer = meas.get("steer", 0.0)
    gt_throttle = meas.get("throttle", 0.0)
    gt_brake = meas.get("brake", 0.0)
    spd = meas.get("speed", 0.0)
    cmd = meas.get("command", 4)

    # Clear all axes
    for ax in axes.flat:
        ax.clear()

    # Main camera image
    ax_img = axes[0, 0]
    img_np = np.array(img_pil)
    ax_img.imshow(img_np)
    ax_img.axis("off")

    # Command overlay on image
    cmd_name = CMD_NAMES.get(cmd, "?")
    cmd_color = CMD_COLORS.get(cmd, "white")
    ax_img.text(img_np.shape[1] // 2, 30, cmd_name, fontsize=14, fontweight="bold",
                color=cmd_color, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # Speed overlay
    ax_img.text(10, img_np.shape[0] - 10,
                f"{spd:.1f} m/s ({spd * 3.6:.0f} km/h)",
                fontsize=10, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    # Steering wheel
    ax_steer = axes[0, 1]
    ax_steer.set_facecolor("#1a1a2e")
    draw_steering_wheel(ax_steer, pred_steer, gt_steer)

    # Control bars
    ax_bars = axes[1, 1]
    ax_bars.set_facecolor("#1a1a2e")
    draw_control_bars(ax_bars, pred_throttle, pred_brake, gt_throttle, gt_brake)

    # Error panel
    ax_err = axes[1, 0]
    ax_err.set_facecolor("#1a1a2e")
    ax_err.axis("off")

    steer_err = abs(pred_steer - gt_steer)
    throttle_err = abs(pred_throttle - gt_throttle)
    brake_err = abs(pred_brake - gt_brake)

    err_color = "#6bcb77" if steer_err < 0.05 else ("#ffd93d" if steer_err < 0.1 else "#ff6b6b")

    text = (
        f"Steer error:    {steer_err:.4f}\n"
        f"Throttle error: {throttle_err:.4f}\n"
        f"Brake error:    {brake_err:.4f}\n"
        f"\nFrame: {frame['frame_id']}"
    )
    ax_err.text(0.5, 0.5, text, fontsize=10, color=err_color,
                ha="center", va="center", family="monospace",
                transform=ax_err.transAxes)

    fig.patch.set_facecolor("#0f0f23")
    fig.suptitle("NeuroDriver BC — Live Prediction",
                 fontsize=13, color="white", fontweight="bold")

    return {
        "pred_steer": pred_steer, "gt_steer": gt_steer,
        "pred_throttle": pred_throttle, "gt_throttle": gt_throttle,
        "pred_brake": pred_brake, "gt_brake": gt_brake,
        "speed": spd, "command": cmd,
    }


def save_video(model, frames, transform, device, output_path, fps=4):
    """Render all frames and save as MP4."""
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("FFmpeg not found. Saving as image strip instead.")
        save_strip(model, frames, transform, device, output_path.replace(".mp4", ".png"))
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    plt.tight_layout(pad=2.0)

    def update(i):
        if i % 10 == 0:
            print(f"  Rendering frame {i + 1}/{len(frames)}...", flush=True)
        render_frame(model, frames[i], transform, device, fig, axes)

    print(f"Rendering {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)

    try:
        writer = FFMpegWriter(fps=fps, metadata={"title": "NeuroDriver BC"})
        anim.save(output_path, writer=writer, dpi=120)
        print(f"Video saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg save failed ({e}), saving as GIF...")
        gif_path = output_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=fps, dpi=80)
        print(f"GIF saved: {gif_path}")

    plt.close()


def save_strip(model, frames, transform, device, output_path, n_frames=16):
    """Save a horizontal strip of frames with predictions."""
    step = max(1, len(frames) // n_frames)
    selected = frames[::step][:n_frames]

    fig, axes_all = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
    fig.patch.set_facecolor("#0f0f23")

    for i, frame in enumerate(selected):
        img_pil = Image.open(frame["rgb_path"]).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        meas = frame["meas"]
        speed = torch.tensor([[meas.get("speed", 0.0)]],
                             dtype=torch.float32, device=device)
        command = torch.tensor([[meas.get("command", 4)]],
                               dtype=torch.long, device=device)

        pred = model(img_tensor, speed, command)

        ps = pred["steer"].item()
        gs = meas.get("steer", 0.0)
        pt = pred["throttle"].item()
        gt = meas.get("throttle", 0.0)
        pb = pred["brake"].item()
        gb = meas.get("brake", 0.0)
        cmd = CMD_NAMES.get(meas.get("command", 4), "?")

        # Camera image
        ax = axes_all[0, i]
        ax.imshow(np.array(img_pil))
        ax.axis("off")
        ax.set_title(f"{cmd}", fontsize=8, color=CMD_COLORS.get(meas.get("command", 4), "white"))

        # Predictions text
        ax2 = axes_all[1, i]
        ax2.set_facecolor("#1a1a2e")
        ax2.axis("off")

        err = abs(ps - gs)
        color = "#6bcb77" if err < 0.03 else ("#ffd93d" if err < 0.08 else "#ff6b6b")

        ax2.text(0.5, 0.5,
                 f"St: {ps:+.3f}/{gs:+.3f}\n"
                 f"Th: {pt:.2f}/{gt:.2f}\n"
                 f"Br: {pb:.2f}/{gb:.2f}\n"
                 f"Spd: {meas.get('speed', 0):.1f}",
                 fontsize=7, color=color, ha="center", va="center",
                 family="monospace", transform=ax2.transAxes)

    plt.suptitle("NeuroDriver BC Predictions (pred/expert)  |  "
                 "Green=accurate, Yellow=close, Red=off",
                 fontsize=11, color="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Strip saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-root", default="data_raw/transfuser")
    parser.add_argument("--route", default=None, help="Route folder name")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    transform = get_val_transforms(tuple(cfg["data"]["image_size"]))

    # Find route
    route = args.route
    if route is None:
        route = find_best_route(args.data_root)
        print(f"Auto-selected route: {route}")

    frames = load_route_frames(args.data_root, route, args.num_frames)
    print(f"Loaded {len(frames)} frames from {route}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_video:
        save_video(model, frames, transform, device,
                   str(out_dir / "bc_driving_demo.mp4"), fps=4)
    else:
        save_strip(model, frames, transform, device,
                   str(out_dir / "bc_driving_strip.png"), n_frames=16)

    # Always save the strip too
    save_strip(model, frames, transform, device,
               str(out_dir / "bc_driving_strip.png"), n_frames=16)

    print("\nDone!")


if __name__ == "__main__":
    main()
"""
Train BEV perception model.

Loss design rationale:
  - road_bce_loss (pos_weight=5): weighted BCE, not focal.
    Focal with alpha=0.75 made negatives near-free at ~8% positive rate,
    so predicting-zero everywhere was optimal. pos_weight=5 forces the
    model to actually predict road in the corridor.
  - nearfield_road_loss: hard constraint — ego is always on road directly
    ahead. Bottom-center BEV cells are always positive. Cannot be dodged.
  - depth_regression_loss: flat-road geometric prior gives direct L1
    supervision on the regression head (below-horizon pixels only).
  - No depth KL loss: fixed-target KL→0 once model memorises geometry,
    leaving the distribution head with no image-conditioned gradient.
    Distribution head is instead shaped by road_loss→splat→lift.

Usage:
    python -m neurodriver.training.train_bev
    python -m neurodriver.training.train_bev --resume checkpoints/bev_model_best.pt
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from neurodriver.data.transforms import get_train_transforms, get_val_transforms
from neurodriver.models.bev_model import BEVDrivingModel
from neurodriver.utils.device import get_device

BEV_X_RANGE = (-25.0, 25.0)
BEV_Y_RANGE = (0.0, 50.0)
BEV_RES     = 0.5
BEV_W = int((BEV_X_RANGE[1] - BEV_X_RANGE[0]) / BEV_RES)   # 100
BEV_H = int((BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) / BEV_RES)   # 100
FEAT_H, FEAT_W = 16, 16


# Label generators 

def generate_road_label(meas):
    """
    Near-field road corridor (0–30m) with explicit far-field zeros.

    Only ego lane ± 0.5 lanes. Confidence decays linearly 1.0→0.5.
    Far-field (>30m) stays 0 — explicit negatives prevent collapse.
    """
    label = np.zeros((BEV_H, BEV_W), dtype=np.float32)

    lane_width = meas.get("ego_lane_width", 3.5)
    dist_to_route = meas.get("distance_ego_to_route", 0.0)
    is_junction = meas.get("is_junction", False)

    half_road_m  = lane_width * (1.5 if not is_junction else 3.0)
    half_road_px = half_road_m / BEV_RES
    lateral_px   = int(dist_to_route / BEV_RES)
    center_x     = int(np.clip(BEV_W // 2 - lateral_px,
                               half_road_px, BEV_W - half_road_px))
    near_limit   = int(30.0 / BEV_RES)   # 60 rows

    for y in range(near_limit):
        conf  = 1.0 - 0.5 * (y / near_limit)
        x_min = max(0, int(center_x - half_road_px))
        x_max = min(BEV_W, int(center_x + half_road_px))
        label[y, x_min:x_max] = conf

    from scipy.ndimage import gaussian_filter
    label = gaussian_filter(label, sigma=1.0)
    return np.clip(label, 0.0, 1.0)


def generate_vehicle_label(meas):
    label = np.zeros((BEV_H, BEV_W), dtype=np.float32)

    obj_dist      = meas.get("speed_reduced_by_obj_distance", 999.0)
    obj_type      = meas.get("speed_reduced_by_obj_type", "none")
    vehicle_haz   = meas.get("vehicle_hazard", False)
    walker_haz    = meas.get("walker_hazard", False)
    ped_dist      = meas.get("dist_to_pedestrian", 999.0)

    def place_blob(dist_m, lateral_m=0.0, size_m=2.5):
        if not (1.0 <= dist_m <= 48.0):
            return
        y_px   = int(dist_m / BEV_RES)
        x_px   = BEV_W // 2 + int(lateral_m / BEV_RES)
        s_px   = max(2, int(size_m / BEV_RES))
        for y in range(max(0, y_px - s_px), min(BEV_H, y_px + s_px)):
            for x in range(max(0, x_px - s_px), min(BEV_W, x_px + s_px)):
                dy = (y - y_px) * BEV_RES
                dx = (x - x_px) * BEV_RES
                label[y, x] = max(label[y, x],
                                  math.exp(-(dy**2 + dx**2) / size_m**2))

    if obj_type in ("car", "vehicle", "truck", "bus", "motorcycle"):
        place_blob(obj_dist, size_m=2.5)
    elif obj_type in ("pedestrian", "walker"):
        place_blob(obj_dist, size_m=1.0)
    if vehicle_haz:
        place_blob(5.0, size_m=3.0)
    if walker_haz:
        place_blob(4.0, lateral_m=1.5, size_m=1.5)
    if ped_dist < 48.0:
        place_blob(ped_dist, lateral_m=2.0, size_m=1.0)
    return label


def generate_geo_depth_label(feat_h=FEAT_H, feat_w=FEAT_W,
                              image_h=256, image_w=256):
    """
    Flat-road geometric depth prior for the feature grid.
    Camera height 1.5m, zero pitch. Sky rows → 0 (no label).
    """
    camera_h = 1.5
    stride_h = image_h / feat_h
    fy       = image_h / 2.0
    horizon  = image_h / 2.0
    depth    = np.zeros((feat_h, feat_w), dtype=np.float32)
    for i in range(feat_h):
        v  = i * stride_h + stride_h / 2
        dv = v - horizon
        if dv > 2.0:
            angle = np.arctan(dv / fy)
            depth[i, :] = np.clip(camera_h / (np.tan(angle) + 1e-6), 2.0, 50.0)
    return depth


GEO_DEPTH_LABEL = torch.tensor(generate_geo_depth_label(), dtype=torch.float32)


# Dataset

class BEVDataset(Dataset):
    def __init__(self, data_root, towns=None, image_size=(256, 256), augment=False):
        self.data_root = Path(data_root)
        self.transform = (get_train_transforms(image_size) if augment
                          else get_val_transforms(image_size))
        self.samples   = []
        self._discover(towns)
        print(f"BEVDataset: {len(self.samples)} samples")
        self._check_label_density()

    def _discover(self, towns):
        for route_dir in sorted(self.data_root.iterdir()):
            if not route_dir.is_dir():
                continue
            if towns and not any(t.lower() in route_dir.name.lower() for t in towns):
                continue
            rgb_dir  = route_dir / "rgb"
            meas_dir = route_dir / "measurements"
            if not rgb_dir.exists() or not meas_dir.exists():
                continue
            rgb_files = {}
            for ext in ("*.jpg", "*.png"):
                for f in rgb_dir.glob(ext):
                    rgb_files[f.stem] = str(f)
            meas_files = {f.stem: str(f) for f in meas_dir.glob("*.json")}
            for fid in sorted(rgb_files.keys() & meas_files.keys()):
                self.samples.append({
                    "rgb_path":  rgb_files[fid],
                    "meas_path": meas_files[fid],
                })

    def _check_label_density(self):
        if not self.samples:
            return
        densities = []
        for s in self.samples[:20]:
            with open(s["meas_path"]) as f:
                meas = json.load(f)
            densities.append((generate_road_label(meas) > 0.3).mean())
        d = float(np.mean(densities))
        status = (" good" if 0.02 <= d <= 0.5
                  else "TOO SPARSE — check ego_lane_width field" if d < 0.02
                  else "TOO DENSE — collapse risk")
        print(f"  Road label density: {d:.3f}  {status}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["rgb_path"]).convert("RGB")
        img = self.transform(img)
        with open(s["meas_path"]) as f:
            meas = json.load(f)
        return {
            "image":          img,
            "road_label":     torch.tensor(generate_road_label(meas),    dtype=torch.float32),
            "vehicle_label":  torch.tensor(generate_vehicle_label(meas), dtype=torch.float32),
        }


# Loss functions

def road_bce_loss(logits, targets, pos_weight=5.0):
    """
    Weighted BCE for road segmentation.
    pos_weight=5 balances ~8% positives vs ~92% background.
    Adjust upward if road head collapses to zero.
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor([pos_weight], device=logits.device),
    )


def nearfield_road_loss(logits):
    """
    Hard constraint: BEV cells directly ahead of ego MUST be road.
    Ego is by definition on the road → bottom-center of BEV is always 1.
    This cannot be dodged by any collapse strategy.

    Region: y ∈ [0, 5m] (10 rows), x ∈ [ego ± 2.5m] (10 cols).
    """
    cx   = logits.shape[-1] // 2
    near = logits[:, 0, :10, cx-5:cx+5]
    return F.binary_cross_entropy_with_logits(near, torch.ones_like(near))


def depth_regression_loss(depth_reg, geo_label_batch):
    """
    Masked L1 on depth regression head vs flat-road geometric prior.
    Mask = below-horizon pixels only (geo_label > 0).
    """
    mask    = (geo_label_batch > 0).float()
    n_valid = mask.sum() + 1e-6
    return ((depth_reg.squeeze(1) - geo_label_batch).abs() * mask).sum() / n_valid


# Diagnostics 

def run_sensitivity_check(model, device, image_size=(256, 256)):
    """Real image vs blank → road head difference. Should be > 0.05."""
    model.eval()
    with torch.no_grad():
        real  = torch.randn(1, 3, *image_size, device=device)
        blank = torch.zeros(1, 3, *image_size, device=device)
        out_r = model(real)
        out_b = model(blank)
        sensitivity = (torch.sigmoid(out_r["road_seg"]) -
                       torch.sigmoid(out_b["road_seg"])).abs().mean().item()
        road_mean   = torch.sigmoid(out_r["road_seg"]).mean().item()
        depth_mean  = out_r["depth_reg"].mean().item()
    model.train()
    return sensitivity, road_mean, depth_mean


# Training 

def train_one_epoch(model, loader, optimizer, device, epoch,
                    geo_depth_label, grad_clip=1.0):
    model.train()
    running = {"total": 0.0, "road": 0.0, "nf_road": 0.0,
               "vehicle": 0.0, "depth_reg": 0.0}
    n = 0

    pbar = tqdm(loader, desc=f"BEV Epoch {epoch}", leave=False)
    for batch in pbar:
        image      = batch["image"].to(device)
        road_gt    = batch["road_label"].to(device)
        vehicle_gt = batch["vehicle_label"].to(device)
        B          = image.shape[0]

        output = model(image)

        road_loss    = road_bce_loss(output["road_seg"].squeeze(1), road_gt,
                                     pos_weight=5.0)
        nf_loss      = nearfield_road_loss(output["road_seg"])
        vehicle_loss = road_bce_loss(output["vehicle_seg"].squeeze(1), vehicle_gt,
                                     pos_weight=20.0)
        geo_batch    = geo_depth_label.unsqueeze(0).expand(B, -1, -1).to(device)
        d_reg_loss   = depth_regression_loss(output["depth_reg"], geo_batch)

        loss = road_loss + 1.0 * nf_loss + 0.3 * vehicle_loss + 0.3 * d_reg_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running["total"]     += loss.item()
        running["road"]      += road_loss.item()
        running["nf_road"]   += nf_loss.item()
        running["vehicle"]   += vehicle_loss.item()
        running["depth_reg"] += d_reg_loss.item()
        n += 1

        pbar.set_postfix({
            "road":  f"{road_loss.item():.4f}",
            "nf":    f"{nf_loss.item():.4f}",
            "d_reg": f"{d_reg_loss.item():.3f}",
        })

    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, geo_depth_label):
    model.eval()
    running = {"total": 0.0, "road": 0.0, "nf_road": 0.0, "depth_reg": 0.0}
    n = 0

    for batch in tqdm(loader, desc="BEV Val", leave=False):
        image   = batch["image"].to(device)
        road_gt = batch["road_label"].to(device)
        B       = image.shape[0]

        output   = model(image)
        r_loss   = road_bce_loss(output["road_seg"].squeeze(1), road_gt)
        nf_loss  = nearfield_road_loss(output["road_seg"])
        geo_b    = geo_depth_label.unsqueeze(0).expand(B, -1, -1).to(device)
        d_loss   = depth_regression_loss(output["depth_reg"], geo_b)

        loss = r_loss + 1.0 * nf_loss + 0.3 * d_loss
        running["total"]     += loss.item()
        running["road"]      += r_loss.item()
        running["nf_road"]   += nf_loss.item()
        running["depth_reg"] += d_loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in running.items()}


def train_bev(
    data_root="data_raw/transfuser",
    train_towns=None,
    val_towns=None,
    epochs=20,
    batch_size=16,
    lr=3e-4,
    num_workers=2,
    checkpoint_dir="checkpoints",
    resume=None,
):
    device = get_device()
    print(f"Device: {device}")

    if train_towns is None:
        train_towns = ["Town01", "Town02", "Town03", "Town04"]
    if val_towns is None:
        val_towns   = ["Town05", "Town06"]

    print("\nLoading BEV datasets...")
    train_ds = BEVDataset(data_root, train_towns, augment=True)
    val_ds   = BEVDataset(data_root, val_towns,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"Train: {len(train_ds)} frames, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} frames, {len(val_loader)} batches")

    print("\nBuilding BEV model...")
    model    = BEVDrivingModel(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  |  lift points/image: {16*16*48:,}")

    # Freeze early backbone (conv1 through layer2)
    for name, param in model.backbone.features.named_parameters():
        if any(name.startswith(f"{i}.") for i in range(6)):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    start_epoch = 1
    best_val    = float("inf")

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val:.4f})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5,
    )
    # Fast-forward scheduler to match resumed epoch so LR is correct
    for _ in range(start_epoch - 1):
        scheduler.step()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining BEV epochs {start_epoch}–{epochs}...\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch, GEO_DEPTH_LABEL
        )
        val_losses = validate(model, val_loader, device, GEO_DEPTH_LABEL)

        scheduler.step()
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_losses['total']:.4f} "
            f"(road={train_losses['road']:.4f} "
            f"nf={train_losses['nf_road']:.4f} "
            f"veh={train_losses['vehicle']:.4f} "
            f"d_reg={train_losses['depth_reg']:.3f}) | "
            f"Val: {val_losses['total']:.4f} | "
            f"LR: {lr_now:.2e} | {elapsed:.0f}s"
        )

        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         best_val,
            }, ckpt_dir / "bev_model_best.pt")
            print(f"  New best saved (val_loss={best_val:.4f})")

        # Diagnostics after checkpoint save
        sensitivity, road_mean, depth_mean = run_sensitivity_check(model, device)
        collapsed = sensitivity < 0.02 or road_mean < 0.02 or road_mean > 0.95
        flag      = "COLLAPSE" if collapsed else "OK!"
        print(f"  Diag: sensitivity={sensitivity:.4f}  "
              f"road_mean={road_mean:.3f}  depth={depth_mean:.1f}m  {flag}")
        if collapsed and epoch >= 3:
            print("  Consider stopping. Run: python scripts/debug_bev.py")

    print(f"\nBEV training complete. Best val: {best_val:.4f}")
    print(f"Visualize: python scripts/visualize_bev.py "
          f"--checkpoint checkpoints/bev_model_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train_bev(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
    )
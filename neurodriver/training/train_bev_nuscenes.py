"""
Train BEV model on nuScenes with real HD map labels.

Key differences from train_bev.py (LEAD dataset):
  - Real drivable_area polygons from HD map (not metadata-derived corridors)
  - Real 3D bounding box vehicle labels (not hazard flag blobs)
  - Real camera intrinsics per sample (not estimated from FOV)
  - Labels vary geometrically every frame — no fixed-prior ceiling

Usage:
    python -m neurodriver.training.train_bev_nuscenes
    python -m neurodriver.training.train_bev_nuscenes --resume checkpoints/bev_nuscenes_best.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurodriver.data.nuscenes_dataset import NuScenesBEVDataset
from neurodriver.models.bev_model import BEVDrivingModel
from neurodriver.utils.device import get_device

DATAROOT = 'data_raw/nuscenes'


# ── Loss functions (same as train_bev.py) ─────────────────────────────────────

def road_bce_loss(logits, targets, pos_weight=5.0):
    return F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor([pos_weight], device=logits.device),
    )


def nearfield_road_loss(logits):
    """Ego is always on road — bottom-center BEV must be road."""
    cx   = logits.shape[-1] // 2
    near = logits[:, 0, :10, cx-5:cx+5]
    return F.binary_cross_entropy_with_logits(near, torch.ones_like(near))


def depth_regression_loss(depth_reg, geo_label_batch):
    mask    = (geo_label_batch > 0).float()
    n_valid = mask.sum() + 1e-6
    return ((depth_reg.squeeze(1) - geo_label_batch).abs() * mask).sum() / n_valid


# ── Geometric depth prior (same as before) ────────────────────────────────────

def make_geo_depth_label(feat_h=16, feat_w=16, image_h=256):
    import numpy as np
    camera_h = 1.5
    stride_h = image_h / feat_h
    fy       = image_h / 2.0
    horizon  = image_h / 2.0
    depth    = np.zeros((feat_h, feat_w), dtype='float32')
    for i in range(feat_h):
        v  = i * stride_h + stride_h / 2
        dv = v - horizon
        if dv > 2.0:
            angle = float(__import__('numpy').arctan(dv / fy))
            depth[i, :] = min(50.0, max(2.0,
                camera_h / (float(__import__('numpy').tan(angle)) + 1e-6)))
    return torch.tensor(depth, dtype=torch.float32)


GEO_DEPTH = make_geo_depth_label()


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_sensitivity_check(model, device):
    model.eval()
    with torch.no_grad():
        real  = torch.randn(1, 3, 256, 256, device=device)
        blank = torch.zeros(1, 3, 256, 256, device=device)
        out_r = model(real)
        out_b = model(blank)
        sens      = (torch.sigmoid(out_r['road_seg']) -
                     torch.sigmoid(out_b['road_seg'])).abs().mean().item()
        road_mean = torch.sigmoid(out_r['road_seg']).mean().item()
        depth_m   = out_r['depth_reg'].mean().item()
    model.train()
    return sens, road_mean, depth_m


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch,
                    geo_depth, grad_clip=1.0):
    model.train()
    running = {'total': 0., 'road': 0., 'nf': 0., 'veh': 0., 'd_reg': 0.}
    n = 0

    pbar = tqdm(loader, desc=f'BEV Epoch {epoch}', leave=False)
    for batch in pbar:
        image   = batch['image'].to(device)
        road_gt = batch['road_label'].to(device)
        veh_gt  = batch['vehicle_label'].to(device)
        B       = image.shape[0]

        out = model(image)

        r_loss  = road_bce_loss(out['road_seg'].squeeze(1), road_gt, pos_weight=1.5)
        nf_loss = nearfield_road_loss(out['road_seg'])
        v_loss  = road_bce_loss(out['vehicle_seg'].squeeze(1), veh_gt, pos_weight=3.0)
        geo_b   = geo_depth.unsqueeze(0).expand(B, -1, -1).to(device)
        d_loss  = depth_regression_loss(out['depth_reg'], geo_b)

        loss = r_loss + 1.0 * nf_loss + 0.3 * v_loss + 0.3 * d_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running['total'] += loss.item()
        running['road']  += r_loss.item()
        running['nf']    += nf_loss.item()
        running['veh']   += v_loss.item()
        running['d_reg'] += d_loss.item()
        n += 1

        pbar.set_postfix({
            'road': f'{r_loss.item():.4f}',
            'nf':   f'{nf_loss.item():.4f}',
        })

    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, geo_depth):
    model.eval()
    running = {'total': 0., 'road': 0., 'nf': 0., 'd_reg': 0.}
    n = 0

    for batch in tqdm(loader, desc='Val', leave=False):
        image   = batch['image'].to(device)
        road_gt = batch['road_label'].to(device)
        B       = image.shape[0]

        out    = model(image)
        r_loss = road_bce_loss(out['road_seg'].squeeze(1), road_gt, pos_weight=1.5)
        nf_l   = nearfield_road_loss(out['road_seg'])
        geo_b  = geo_depth.unsqueeze(0).expand(B, -1, -1).to(device)
        d_loss = depth_regression_loss(out['depth_reg'], geo_b)

        loss = r_loss + 1.0 * nf_l + 0.3 * d_loss
        running['total'] += loss.item()
        running['road']  += r_loss.item()
        running['nf']    += nf_l.item()
        running['d_reg'] += d_loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in running.items()}


def train_bev_nuscenes(
    dataroot=DATAROOT,
    epochs=30,
    batch_size=8,       # nuScenes mini has 324 train samples — small batches
    lr=3e-4,
    num_workers=2,
    checkpoint_dir='checkpoints',
    resume=None,
):
    device = get_device()
    print(f'Device: {device}')

    print('\nLoading nuScenes datasets...')
    train_ds = NuScenesBEVDataset(dataroot, split='train', augment=True)
    val_ds   = NuScenesBEVDataset(dataroot, split='val',   augment=False)

    # nuScenes mini is small — drop_last=False to use all samples
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f'Train: {len(train_ds)} samples, {len(train_loader)} batches')
    print(f'Val:   {len(val_ds)} samples, {len(val_loader)} batches')

    print('\nBuilding BEV model...')
    model = BEVDrivingModel(pretrained=True).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Freeze early backbone (conv1 through layer2)
    for name, param in model.backbone.features.named_parameters():
        if any(name.startswith(f'{i}.') for i in range(6)):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {trainable:,}')

    optimizer   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    start_epoch = 1
    best_val    = float('inf')

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt['val_loss']
        print(f'Resumed from epoch {ckpt["epoch"]} (val_loss={best_val:.4f})')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nTraining BEV on nuScenes, epochs {start_epoch}–{epochs}...\n')

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch, GEO_DEPTH
        )
        val_losses = validate(model, val_loader, device, GEO_DEPTH)

        scheduler.step(val_losses['total'])
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(
            f'Epoch {epoch:3d}/{epochs} | '
            f'Train: {train_losses["total"]:.4f} '
            f'(road={train_losses["road"]:.4f} '
            f'nf={train_losses["nf"]:.4f} '
            f'veh={train_losses["veh"]:.4f} '
            f'd_reg={train_losses["d_reg"]:.3f}) | '
            f'Val: {val_losses["total"]:.4f} | '
            f'LR: {lr_now:.2e} | {elapsed:.0f}s'
        )

        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         best_val,
            }, ckpt_dir / 'bev_nuscenes_best.pt')
            print(f'  New best saved (val_loss={best_val:.4f})')

        # Save every 5 epochs so we never lose good checkpoints
        if epoch % 5 == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         val_losses['total'],
            }, ckpt_dir / f'bev_nuscenes_epoch_{epoch:03d}.pt')

        # Diagnostics
        sens, road_mean, depth_m = run_sensitivity_check(model, device)
        collapsed = sens < 0.02 or road_mean < 0.02 or road_mean > 0.95
        flag      = 'COLLAPSE' if collapsed else 'OK'
        print(f'  Diag: sensitivity={sens:.4f}  '
              f'road_mean={road_mean:.3f}  depth={depth_m:.1f}m  {flag}')

    print(f'\nDone. Best val: {best_val:.4f}')
    print(f'Visualize: python scripts/visualize_bev.py '
          f'--checkpoint checkpoints/bev_nuscenes_best.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--resume',     type=str,   default=None)
    args = parser.parse_args()

    train_bev_nuscenes(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
    )
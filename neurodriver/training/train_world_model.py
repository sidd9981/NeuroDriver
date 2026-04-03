"""
World Model Training v2 — with DreamerV3-style free bits to prevent KL collapse.

Changes from v1:
  1. Free bits (free_nats=1.0): KL below 1.0 nat is not penalized,
     preventing the posterior from collapsing onto the prior.
  2. Uses compute_reward_v2 (Roach-style) via updated SequenceDataset.
  3. grad_clip lowered to 10.0 (v1 had 100.0 which is basically no clipping).

Usage:
    python -m neurodriver.training.train_world_model
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurodriver.data.sequence_dataset import SequenceDataset
from neurodriver.models.world_model import WorldModel
from neurodriver.utils.device import get_device


def compute_kl_loss(priors, posteriors, free_nats=1.0):
    """
    KL divergence with DreamerV3-style free bits.

    free_nats: minimum TOTAL KL (summed across dimensions).
    If KL is already above this, no effect. If below, clamp to floor.
    This prevents collapse without dominating the loss.
    """
    T = len(priors)
    kl_total = torch.tensor(0.0, device=priors[0][0].device)

    for t in range(T):
        prior_mean, prior_std = priors[t]
        post_mean, post_std = posteriors[t]

        prior_dist = Normal(prior_mean, prior_std)
        post_dist = Normal(post_mean, post_std)

        # Sum across stoch_dim first, THEN apply floor to the total
        kl_sum = kl_divergence(post_dist, prior_dist).sum(dim=-1)  # (B,)
        kl_clamped = torch.clamp(kl_sum, min=free_nats)  # floor on total
        kl_total = kl_total + kl_clamped.mean()

    return kl_total / max(T, 1)


def train_one_epoch(model, loader, optimizer, device, epoch, grad_clip=10.0,
                    kl_weight=1.0, free_nats=1.0):
    model.train()

    running = {"total": 0.0, "recon": 0.0, "reward": 0.0, "kl": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"WM Epoch {epoch}", leave=False)

    for batch in pbar:
        images = batch["images"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        output = model.observe_sequence(images, actions)

        recon_loss = F.mse_loss(output["decoded"], output["embeds"].detach())
        reward_loss = F.mse_loss(output["rewards"], rewards)
        kl_loss = compute_kl_loss(output["priors"], output["posteriors"],
                                  free_nats=free_nats)

        loss = recon_loss + reward_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running["total"] += loss.item()
        running["recon"] += recon_loss.item()
        running["reward"] += reward_loss.item()
        running["kl"] += kl_loss.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "rew": f"{reward_loss.item():.4f}",
            "kl": f"{kl_loss.item():.3f}",
        })

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, kl_weight=1.0, free_nats=1.0):
    model.eval()

    running = {"total": 0.0, "recon": 0.0, "reward": 0.0, "kl": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="WM Val", leave=False):
        images = batch["images"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        output = model.observe_sequence(images, actions)

        recon_loss = F.mse_loss(output["decoded"], output["embeds"].detach())
        reward_loss = F.mse_loss(output["rewards"], rewards)
        kl_loss = compute_kl_loss(output["priors"], output["posteriors"],
                                  free_nats=free_nats)

        loss = recon_loss + reward_loss + kl_weight * kl_loss

        running["total"] += loss.item()
        running["recon"] += recon_loss.item()
        running["reward"] += reward_loss.item()
        running["kl"] += kl_loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def train_world_model(
    data_root: str = "data_raw/transfuser",
    train_towns: list = None,
    val_towns: list = None,
    seq_len: int = 16,
    batch_size: int = 8,
    epochs: int = 30,
    lr: float = 3e-4,
    kl_weight: float = 1.0,
    free_nats: float = 1.0,
    grad_clip: float = 10.0,
    num_workers: int = 2,
    checkpoint_dir: str = "checkpoints",
):
    device = get_device()
    print(f"Device: {device}")
    print(f"KL weight: {kl_weight}, free_nats: {free_nats}")

    if train_towns is None:
        train_towns = ["Town01", "Town02", "Town03", "Town04"]
    if val_towns is None:
        val_towns = ["Town05", "Town06"]

    print("\nLoading sequence datasets (with Roach-style reward v2)...")
    train_dataset = SequenceDataset(
        data_root=data_root, towns=train_towns, seq_len=seq_len, stride=4,
    )
    val_dataset = SequenceDataset(
        data_root=data_root, towns=val_towns, seq_len=seq_len, stride=8,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    print(f"Train: {len(train_dataset)} sequences, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} sequences, {len(val_loader)} batches")

    # Quick reward stats check
    sample = train_dataset[0]
    r = sample["rewards"]
    print(f"Sample reward stats: mean={r.mean():.3f}, std={r.std():.3f}, "
          f"min={r.min():.3f}, max={r.max():.3f}")

    print("\nBuilding world model...")
    model = WorldModel(
        stoch_dim=64, deter_dim=256, hidden_dim=256, action_dim=3, embed_dim=256,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"\nTraining world model v2 for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            grad_clip=grad_clip, kl_weight=kl_weight, free_nats=free_nats,
        )

        val_losses = validate(
            model, val_loader, device, kl_weight=kl_weight, free_nats=free_nats,
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_losses['total']:.4f} "
            f"(r={train_losses['recon']:.4f} "
            f"rew={train_losses['reward']:.4f} "
            f"kl={train_losses['kl']:.3f}) | "
            f"Val: {val_losses['total']:.4f} "
            f"(kl={val_losses['kl']:.3f}) | "
            f"LR: {lr_now:.2e} | {elapsed:.0f}s"
        )

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_path = ckpt_dir / "world_model_v2_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "free_nats": free_nats,
                "kl_weight": kl_weight,
            }, save_path)
            print(f"  New best saved (val_loss={best_val_loss:.4f})")

        if epoch % 10 == 0:
            save_path = ckpt_dir / f"world_model_v2_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_losses["total"],
            }, save_path)

    print(f"\nWorld model v2 training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {ckpt_dir / 'world_model_v2_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    args = parser.parse_args()

    train_world_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        kl_weight=args.kl_weight,
        free_nats=args.free_nats,
        grad_clip=args.grad_clip,
    )
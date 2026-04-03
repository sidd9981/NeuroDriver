"""
Behavioral Cloning Training Loop — Phase 1

This is straightforward supervised learning:
    Expert drives -> we record (image, controls) pairs -> we train a model
    to predict the controls from the images.

The training loop is intentionally simple and readable.
No fancy frameworks, no abstractions — just PyTorch.

Usage:
    python -m neurodriver.training.train_bc

    Or with config overrides:
    python -m neurodriver.training.train_bc --config configs/bc.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurodriver.data.dataset import build_dataset
from neurodriver.models.e2e_model import DrivingModel, build_model
from neurodriver.utils.device import get_device


#  Loss Function 

class DrivingLoss(nn.Module):
    """
    Multi-task loss for behavioral cloning.
    
    Total loss = control_weight * control_loss
               + speed_weight * speed_loss
               + waypoint_weight * waypoint_loss
    
    Control loss: L1 loss on [steer, throttle, brake]
        Why L1 instead of MSE? L1 is more robust to outlier frames
        (e.g., sudden emergency brakes). TCP uses L1.
    
    Speed loss: L1 on predicted vs actual speed.
        Forces the model to understand how fast it's going.
    
    Waypoint loss: L1 on predicted waypoints (when available).
        Forces spatial understanding of the road layout.
    """
    
    def __init__(
        self,
        control_weight: float = 1.0,
        speed_weight: float = 0.5,
        waypoint_weight: float = 0.5,
    ):
        super().__init__()
        self.control_weight = control_weight
        self.speed_weight = speed_weight
        self.waypoint_weight = waypoint_weight
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            pred: Model output dict with steer, throttle, brake, pred_speed, pred_waypoints.
            target: Ground truth dict with steer, throttle, brake, speed, target_point.
        
        Returns:
            Dict with 'total' loss and individual components for logging.
        """
        # Control loss — the main objective
        steer_loss = self.l1(pred["steer"], target["steer"])
        throttle_loss = self.l1(pred["throttle"], target["throttle"])
        brake_loss = self.l1(pred["brake"], target["brake"])
        control_loss = steer_loss + throttle_loss + brake_loss
        
        # Speed prediction loss — auxiliary
        speed_loss = self.l1(pred["pred_speed"], target["speed"])
        
        # Total weighted loss
        total = (
            self.control_weight * control_loss
            + self.speed_weight * speed_loss
        )
        
        return {
            "total": total,
            "control": control_loss.detach(),
            "steer": steer_loss.detach(),
            "throttle": throttle_loss.detach(),
            "brake": brake_loss.detach(),
            "speed": speed_loss.detach(),
        }


#  Training Loop 

def train_one_epoch(
    model: DrivingModel,
    loader: DataLoader,
    criterion: DrivingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """
    Train for one epoch.
    
    Returns dict of average losses for logging.
    """
    model.train()
    
    # Accumulate losses
    running = {k: 0.0 for k in ["total", "control", "steer", "throttle", "brake", "speed"]}
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch in pbar:
        # Move to device
        image = batch["image"].to(device)
        speed = batch["speed"].to(device)
        command = batch["command"].to(device)
        
        targets = {
            "steer": batch["steer"].to(device),
            "throttle": batch["throttle"].to(device),
            "brake": batch["brake"].to(device),
            "speed": batch["speed"].to(device),
        }
        
        # Forward pass
        pred = model(image, speed, command)
        
        # Compute loss
        losses = criterion(pred, targets)
        
        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()
        
        # Gradient clipping — prevents exploding gradients
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate for logging
        for k in running:
            running[k] += losses[k].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "steer": f"{losses['steer'].item():.4f}",
        })
    
    # Average over epoch
    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: DrivingModel,
    loader: DataLoader,
    criterion: DrivingLoss,
    device: torch.device,
) -> dict[str, float]:
    """
    Validate on held-out data.
    
    Returns dict of average losses.
    """
    model.eval()
    
    running = {k: 0.0 for k in ["total", "control", "steer", "throttle", "brake", "speed"]}
    n_batches = 0
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        image = batch["image"].to(device)
        speed = batch["speed"].to(device)
        command = batch["command"].to(device)
        
        targets = {
            "steer": batch["steer"].to(device),
            "throttle": batch["throttle"].to(device),
            "brake": batch["brake"].to(device),
            "speed": batch["speed"].to(device),
        }
        
        pred = model(image, speed, command)
        losses = criterion(pred, targets)
        
        for k in running:
            running[k] += losses[k].item()
        n_batches += 1
    
    return {k: v / max(n_batches, 1) for k, v in running.items()}


#  Main Training Script 

def train(cfg: dict):
    """
    Full training pipeline.
    
    Args:
        cfg: Configuration dictionary (loaded from YAML).
    """
    #  Setup 
    device = get_device(cfg.get("device") if cfg.get("device") != "auto" else None)
    print(f"Device: {device}")
    
    torch.manual_seed(cfg.get("seed", 42))
    
    #  Data 
    print("\nLoading datasets")
    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")
    
    train_cfg = cfg["training"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    #  Model 
    print("\nBuilding model")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    #  Optimizer 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    
    #  LR Scheduler 
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
            eta_min=train_cfg.get("min_lr", 1e-6),
        )
    
    #  Loss 
    loss_cfg = train_cfg["loss"]
    criterion = DrivingLoss(
        control_weight=loss_cfg["control_weight"],
        speed_weight=loss_cfg["speed_weight"],
        waypoint_weight=loss_cfg.get("waypoint_weight", 0.0),
    )
    
    #  Checkpoint directory 
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    #  Training Loop 
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print("\n")
    
    best_val_loss = float("inf")
    
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_clip=train_cfg.get("grad_clip", 1.0),
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        
        # Print summary
        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            f"Train loss: {train_losses['total']:.4f} | "
            f"Val loss: {val_losses['total']:.4f} | "
            f"Steer: {val_losses['steer']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg,
            }, save_path)
            print(f"  New best model saved (val_loss={best_val_loss:.4f})")
        
        # Periodic checkpoint
        if epoch % train_cfg.get("save_every", 5) == 0:
            save_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_losses["total"],
                "config": cfg,
            }, save_path)
    
    print("\n")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {ckpt_dir / 'best.pt'}")


#  Entry Point 

def load_config(path: str | None = None) -> dict:
    """Load config from YAML file, or return defaults for testing."""
    if path is not None and Path(path).exists():
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)
    
    # Default config for quick testing with dummy data
    return {
        "data": {
            "dataset_type": "transfuser",
            "data_root": "data_raw/transfuser",
            "train_towns": ["Town01", "Town02", "Town03", "Town04", "Town06"],
            "val_towns": ["Town05"],
            "image_size": [256, 256],
            "seq_len": 1,
            "augment": True,
            "augment_config": {
                "color_jitter": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
            },
        },
        "model": {
            "backbone": "resnet34",
            "pretrained": True,
            "feature_dim": 512,
            "speed_embed_dim": 64,
            "command_embed_dim": 64,
            "num_commands": 4,
            "temporal": {"enabled": False},
            "policy": {
                "hidden_dims": [512, 256],
                "dropout": 0.1,
                "output_dim": 3,
            },
        },
        "training": {
            "epochs": 50,
            "batch_size": 64,
            "num_workers": 4,
            "optimizer": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "min_lr": 1e-6,
            "grad_clip": 1.0,
            "loss": {
                "control_weight": 1.0,
                "speed_weight": 0.5,
                "waypoint_weight": 0.0,
            },
            "save_every": 5,
            "checkpoint_dir": "checkpoints",
        },
        "logging": {
            "use_wandb": False,
            "project_name": "neurodriver",
            "log_every": 50,
        },
        "device": "auto",
        "seed": 42,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC model")
    parser.add_argument("--config", type=str, default="configs/bc.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train(cfg)
"""
RL Fine-Tuning of BC Policy v2.

Changes from v1:
  1. Uses world_model_v2 (free bits, Roach reward)
  2. Imagines with DIFFERENT actions per timestep (not repeated)
     by running the BC model on each imagined state
  3. Proper advantage normalization
  4. Steering-preservation loss weighted higher

Usage:
    python -m neurodriver.training.train_rl_finetune
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neurodriver.data.dataset import TransFuserDataset
from neurodriver.models.e2e_model import DrivingModel
from neurodriver.models.world_model import WorldModel
from neurodriver.utils.device import get_device


class ValueHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class BCLatentPolicy(nn.Module):
    """
    Wraps BC model's control head to act in world model latent space.

    During imagination, we don't have real images — only latent states.
    This small adapter maps latent state -> action, initialized from BC
    control head weights where possible.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.steer_head = nn.Linear(hidden_dim, 1)
        self.throttle_head = nn.Linear(hidden_dim, 1)
        self.brake_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        h = self.net(state)
        steer = torch.tanh(self.steer_head(h))
        throttle = torch.sigmoid(self.throttle_head(h))
        brake = torch.sigmoid(self.brake_head(h))
        return torch.cat([steer, throttle, brake], dim=-1)


def load_bc_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DrivingModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False,
        feature_dim=cfg["model"]["feature_dim"],
        speed_embed_dim=cfg["model"]["speed_embed_dim"],
        command_embed_dim=cfg["model"]["command_embed_dim"],
        num_commands=cfg["model"]["num_commands"],
        hidden_dims=cfg["model"]["policy"]["hidden_dims"],
        dropout=cfg["model"]["policy"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def load_world_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    wm = WorldModel(
        stoch_dim=64, deter_dim=256, hidden_dim=256, action_dim=3, embed_dim=256
    ).to(device)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    return wm


def get_bc_features_and_actions(bc_model, image, speed, command):
    """Get BC actions and fused features (for value head)."""
    img_features = bc_model.backbone(image)
    speed_emb = bc_model.speed_encoder(speed)
    cmd_emb = bc_model.command_encoder(command)
    fused = torch.cat([img_features, speed_emb, cmd_emb], dim=1)
    controls = bc_model.control_head(fused)
    action = torch.cat([
        controls["steer"], controls["throttle"], controls["brake"],
    ], dim=-1)
    return action, fused, controls


def imagine_with_policy(world_model, latent_policy, start_image, start_action,
                        device, horizon=10):
    """
    Imagine forward using the latent policy (not repeated actions).

    Each step: latent_policy picks action from current state,
    world model predicts next state + reward.
    """
    B = start_image.shape[0]

    embed = world_model.encoder(start_image)
    state = world_model.rssm.initial_state(B, device)
    state = world_model.rssm.observe_step(state, start_action, embed)

    all_rewards = []
    all_continues = []
    all_actions = []

    for t in range(horizon):
        full_state = world_model.rssm.get_full_state(state)
        action = latent_policy(full_state)
        all_actions.append(action)

        state = world_model.rssm.imagine_step(state, action)
        next_full = world_model.rssm.get_full_state(state)

        r = world_model.reward_model(next_full)
        c = torch.sigmoid(world_model.continue_model(next_full))
        all_rewards.append(r)
        all_continues.append(c)

    rewards = torch.stack(all_rewards, dim=1)     # (B, H, 1)
    continues = torch.stack(all_continues, dim=1)  # (B, H, 1)
    actions = torch.stack(all_actions, dim=1)      # (B, H, 3)

    return rewards, continues, actions


def compute_returns(rewards, continues, gamma=0.99):
    B, H, _ = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, 1, device=rewards.device)
    for t in reversed(range(H)):
        running = rewards[:, t] + gamma * continues[:, t] * running
        returns[:, t] = running
    return returns


def train_rl_finetune(
    bc_checkpoint="checkpoints/best.pt",
    wm_checkpoint="checkpoints/world_model_v2_best.pt",
    data_root="data_raw/transfuser",
    train_towns=None,
    num_updates=2000,
    batch_size=32,
    imagine_horizon=10,
    policy_lr=3e-5,
    value_lr=1e-4,
    bc_reg_weight=0.3,
    steer_sup_weight=1.0,
    grad_clip=0.5,
    log_every=50,
    checkpoint_dir="checkpoints",
):
    device = get_device()
    print(f"Device: {device}")

    if train_towns is None:
        train_towns = ["Town01", "Town02", "Town03", "Town04"]

    # Load models
    print("\nLoading BC model...")
    bc_model, bc_cfg = load_bc_model(bc_checkpoint, device)

    bc_original = copy.deepcopy(bc_model)
    bc_original.eval()
    for p in bc_original.parameters():
        p.requires_grad = False

    print("Loading world model v2...")
    world_model = load_world_model(wm_checkpoint, device)

    # Freeze BC backbone
    for p in bc_model.backbone.parameters():
        p.requires_grad = False

    # Build latent policy for imagination
    state_dim = world_model.rssm.full_state_dim
    latent_policy = BCLatentPolicy(state_dim, hidden_dim=256).to(device)

    # Value head on BC fused features
    fused_dim = (
        bc_cfg["model"]["feature_dim"]
        + bc_cfg["model"]["speed_embed_dim"]
        + bc_cfg["model"]["command_embed_dim"]
    )
    value_head = ValueHead(fused_dim).to(device)

    # Optimizers
    bc_policy_params = (
        list(bc_model.control_head.parameters())
        + list(bc_model.speed_encoder.parameters())
        + list(bc_model.command_encoder.parameters())
    )
    latent_params = list(latent_policy.parameters())

    policy_optimizer = torch.optim.Adam(
        bc_policy_params + latent_params, lr=policy_lr
    )
    value_optimizer = torch.optim.Adam(value_head.parameters(), lr=value_lr)

    bc_trainable = sum(p.numel() for p in bc_policy_params)
    latent_trainable = sum(p.numel() for p in latent_params)
    print(f"BC trainable: {bc_trainable:,}, Latent policy: {latent_trainable:,}")

    # Data
    print("\nLoading training data...")
    dataset = TransFuserDataset(
        data_root=data_root, towns=train_towns,
        image_size=tuple(bc_cfg["data"]["image_size"]), augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )
    loader_iter = iter(loader)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRL fine-tuning v2: {num_updates} updates, horizon={imagine_horizon}")
    print(f"Policy LR: {policy_lr}, BC reg: {bc_reg_weight}, "
          f"Steer sup: {steer_sup_weight}\n")

    metrics_history = []
    best_reward = float("-inf")

    for update in range(1, num_updates + 1):
        bc_model.train()
        latent_policy.train()
        value_head.train()

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        image = batch["image"].to(device)
        speed = batch["speed"].to(device)
        command = batch["command"].to(device)
        gt_steer = batch["steer"].to(device)
        gt_throttle = batch["throttle"].to(device)
        gt_brake = batch["brake"].to(device)

        # BC actions on real images
        action, fused, controls = get_bc_features_and_actions(
            bc_model, image, speed, command
        )

        # Original BC actions for regularization
        with torch.no_grad():
            _, _, orig_controls = get_bc_features_and_actions(
                bc_original, image, speed, command
            )

        # Imagine forward with latent policy (proper per-step actions)
        imagined_rewards, imagined_continues, _ = imagine_with_policy(
            world_model, latent_policy, image, action, device,
            horizon=imagine_horizon,
        )
        returns = compute_returns(imagined_rewards, imagined_continues)
        value_target = returns[:, 0].detach()

        # Update value head
        value_optimizer.zero_grad()
        value_pred = value_head(fused.detach())
        value_loss = F.mse_loss(value_pred, value_target)
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
        value_optimizer.step()

        # Recompute with fresh graph for policy gradient
        action2, fused2, controls2 = get_bc_features_and_actions(
            bc_model, image, speed, command
        )

        imagined_rewards2, imagined_continues2, _ = imagine_with_policy(
            world_model, latent_policy, image, action2, device,
            horizon=imagine_horizon,
        )
        returns2 = compute_returns(imagined_rewards2, imagined_continues2)

        with torch.no_grad():
            baseline = value_head(fused2)
        advantage = returns2[:, 0] - baseline

        # Normalize advantages (critical for stable RL)
        if advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy loss: maximize advantage-weighted return
        policy_loss = -(advantage * returns2[:, 0]).mean()

        # BC regularization
        bc_reg = (
            F.mse_loss(controls2["steer"], orig_controls["steer"].detach())
            + F.mse_loss(controls2["throttle"], orig_controls["throttle"].detach())
            + F.mse_loss(controls2["brake"], orig_controls["brake"].detach())
        )

        # Steering supervision (BC is good at this, preserve it)
        steer_supervised = F.l1_loss(controls2["steer"], gt_steer)

        total_loss = (
            policy_loss
            + bc_reg_weight * bc_reg
            + steer_sup_weight * steer_supervised
        )

        policy_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(bc_policy_params + latent_params, grad_clip)
        policy_optimizer.step()

        # Metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "bc_reg": bc_reg.item(),
            "steer_sup": steer_supervised.item(),
            "reward_mean": imagined_rewards.mean().item(),
            "advantage_std": advantage.std().item(),
            "steer_mae": (controls["steer"] - gt_steer).abs().mean().item(),
            "throttle_mae": (controls["throttle"] - gt_throttle).abs().mean().item(),
            "brake_mae": (controls["brake"] - gt_brake).abs().mean().item(),
        }
        metrics_history.append(metrics)

        if update % log_every == 0:
            recent = metrics_history[-log_every:]
            avg = {k: sum(m[k] for m in recent) / len(recent) for k in recent[0]}

            print(
                f"Update {update:4d}/{num_updates} | "
                f"Rew: {avg['reward_mean']:+.4f} | "
                f"St: {avg['steer_mae']:.4f} | "
                f"Th: {avg['throttle_mae']:.4f} | "
                f"Br: {avg['brake_mae']:.4f} | "
                f"BC: {avg['bc_reg']:.4f} | "
                f"VL: {avg['value_loss']:.4f} | "
                f"Adv: {avg['advantage_std']:.3f}"
            )

            if avg["reward_mean"] > best_reward:
                best_reward = avg["reward_mean"]
                torch.save({
                    "update": update,
                    "model_state_dict": bc_model.state_dict(),
                    "latent_policy_state_dict": latent_policy.state_dict(),
                    "reward_mean": best_reward,
                    "config": bc_cfg,
                }, ckpt_dir / "bc_rl_v2_best.pt")
                print(f"  New best saved (reward={best_reward:.4f})")

    # Final save
    torch.save({
        "update": num_updates,
        "model_state_dict": bc_model.state_dict(),
        "latent_policy_state_dict": latent_policy.state_dict(),
        "config": bc_cfg,
        "metrics_history": metrics_history,
    }, ckpt_dir / "bc_rl_v2_final.pt")

    print(f"\nRL fine-tuning v2 complete. Best reward: {best_reward:.4f}")
    print(f"Saved to: {ckpt_dir / 'bc_rl_v2_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--policy-lr", type=float, default=3e-5)
    parser.add_argument("--bc-reg", type=float, default=0.3)
    parser.add_argument("--steer-sup", type=float, default=1.0)
    parser.add_argument("--bc-checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--wm-checkpoint", default="checkpoints/world_model_v2_best.pt")
    args = parser.parse_args()

    train_rl_finetune(
        bc_checkpoint=args.bc_checkpoint,
        wm_checkpoint=args.wm_checkpoint,
        num_updates=args.num_updates,
        batch_size=args.batch_size,
        imagine_horizon=args.horizon,
        policy_lr=args.policy_lr,
        bc_reg_weight=args.bc_reg,
        steer_sup_weight=args.steer_sup,
    )
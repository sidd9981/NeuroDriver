"""
Compare BC policy vs RL-trained policy on real validation data.

The RL actor operates in latent space, so we:
  1. Encode real images through the world model encoder
  2. Run the RSSM to get latent states
  3. Run both the RL actor and the BC model on the same frames
  4. Compare their predictions against expert ground truth

Usage:
    python scripts/eval_rl_vs_bc.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurodriver.data.sequence_dataset import SequenceDataset, compute_reward
from neurodriver.models.e2e_model import DrivingModel
from neurodriver.models.world_model import WorldModel
from neurodriver.utils.device import get_device


def load_bc_model(checkpoint_path, device):
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
    return model


def load_world_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wm = WorldModel(
        stoch_dim=64, deter_dim=256, hidden_dim=256, action_dim=3, embed_dim=256
    ).to(device)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()
    return wm


def load_rl_actor(checkpoint_path, state_dim, device):
    from neurodriver.training.train_rl import ImagineActor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = ImagineActor(state_dim, action_dim=3, hidden_dim=256).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


@torch.no_grad()
def evaluate_both(bc_model, world_model, rl_actor, loader, device, max_seqs=100):
    """Run both models on the same data and collect metrics."""

    bc_results = {"steer": [], "throttle": [], "brake": [], "rewards": []}
    rl_results = {"steer": [], "throttle": [], "brake": [], "rewards": []}
    gt_results = {"steer": [], "throttle": [], "brake": [], "rewards": []}

    count = 0
    for batch in loader:
        if count >= max_seqs:
            break

        images = batch["images"].to(device)    # (B, T, 3, H, W)
        actions = batch["actions"].to(device)   # (B, T, 3)
        rewards = batch["rewards"]              # (B, T, 1)
        speeds = batch["speeds"].to(device)     # (B, T, 1)

        B, T = images.shape[:2]

        # --- BC model predictions (frame by frame) ---
        for t in range(T):
            bc_out = bc_model(
                images[:, t],
                speeds[:, t],
                torch.full((B, 1), 4, dtype=torch.long, device=device),  # follow lane
            )
            bc_results["steer"].append(bc_out["steer"].cpu())
            bc_results["throttle"].append(bc_out["throttle"].cpu())
            bc_results["brake"].append(bc_out["brake"].cpu())

        # --- RL actor predictions (through world model latent space) ---
        state = world_model.rssm.initial_state(B, device)
        for t in range(T):
            embed = world_model.encoder(images[:, t])
            state = world_model.rssm.observe_step(state, actions[:, t], embed)
            full_state = world_model.rssm.get_full_state(state)

            rl_action = rl_actor.get_action(full_state, deterministic=True)
            rl_results["steer"].append(rl_action[:, 0:1].cpu())
            rl_results["throttle"].append(rl_action[:, 1:2].cpu())
            rl_results["brake"].append(rl_action[:, 2:3].cpu())

        # --- Ground truth ---
        for t in range(T):
            gt_results["steer"].append(actions[:, t, 0:1].cpu())
            gt_results["throttle"].append(actions[:, t, 1:2].cpu())
            gt_results["brake"].append(actions[:, t, 2:3].cpu())
            gt_results["rewards"].append(rewards[:, t])

        count += B

    # Flatten
    for key in ["steer", "throttle", "brake"]:
        bc_results[key] = torch.cat(bc_results[key]).squeeze().numpy()
        rl_results[key] = torch.cat(rl_results[key]).squeeze().numpy()
        gt_results[key] = torch.cat(gt_results[key]).squeeze().numpy()
    gt_results["rewards"] = torch.cat(gt_results["rewards"]).squeeze().numpy()

    return bc_results, rl_results, gt_results


def print_comparison(bc, rl, gt):
    print("\n" + "=" * 65)
    print("  BC vs RL Policy Comparison on Real Validation Data")
    print("=" * 65)
    print(f"  {'Metric':<25} {'BC':>10} {'RL':>10} {'Winner':>10}")
    print("-" * 65)

    for ctrl in ["steer", "throttle", "brake"]:
        bc_mae = np.abs(bc[ctrl] - gt[ctrl]).mean()
        rl_mae = np.abs(rl[ctrl] - gt[ctrl]).mean()
        winner = "BC" if bc_mae < rl_mae else "RL"
        if abs(bc_mae - rl_mae) < 0.001:
            winner = "Tie"
        print(f"  {ctrl + ' MAE':<25} {bc_mae:>10.4f} {rl_mae:>10.4f} {winner:>10}")

    # Correlation
    for ctrl in ["steer", "throttle"]:
        bc_corr = np.corrcoef(bc[ctrl], gt[ctrl])[0, 1]
        rl_corr = np.corrcoef(rl[ctrl], gt[ctrl])[0, 1]
        winner = "BC" if bc_corr > rl_corr else "RL"
        print(f"  {ctrl + ' correlation':<25} {bc_corr:>10.3f} {rl_corr:>10.3f} {winner:>10}")

    # Compute simulated rewards for each policy
    bc_reward = compute_policy_reward(bc, gt)
    rl_reward = compute_policy_reward(rl, gt)
    gt_reward = gt["rewards"].mean()

    print("-" * 65)
    print(f"  {'Mean driving reward':<25} {bc_reward:>10.4f} {rl_reward:>10.4f} {'BC' if bc_reward > rl_reward else 'RL':>10}")
    print(f"  {'Expert reward':<25} {gt_reward:>10.4f}")
    print("=" * 65)


def compute_policy_reward(policy_results, gt):
    """Compute approximate driving reward from a policy's outputs."""
    n = len(policy_results["steer"])
    rewards = []
    for i in range(n):
        meas = {
            "speed": float(gt.get("speeds", np.zeros(n))[i]) if "speeds" in gt else 5.0,
            "steer": float(policy_results["steer"][i]),
            "brake": float(policy_results["brake"][i]),
        }
        prev = None
        if i > 0:
            prev = {
                "steer": float(policy_results["steer"][i - 1]),
            }
        rewards.append(compute_reward(meas, prev))
    return np.mean(rewards)


def plot_comparison(bc, rl, gt, save_path="checkpoints/bc_vs_rl_comparison.png"):
    n = min(200, len(gt["steer"]))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    x = range(n)

    # Steering
    axes[0].plot(x, gt["steer"][:n], "b-", alpha=0.5, label="Expert", linewidth=1)
    axes[0].plot(x, bc["steer"][:n], "r-", alpha=0.7, label="BC", linewidth=1)
    axes[0].plot(x, rl["steer"][:n], "g-", alpha=0.7, label="RL", linewidth=1)
    axes[0].set_ylabel("Steering")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("BC vs RL Policy: Predicted Controls Over Time")

    # Throttle
    axes[1].plot(x, gt["throttle"][:n], "b-", alpha=0.5, label="Expert", linewidth=1)
    axes[1].plot(x, bc["throttle"][:n], "r-", alpha=0.7, label="BC", linewidth=1)
    axes[1].plot(x, rl["throttle"][:n], "g-", alpha=0.7, label="RL", linewidth=1)
    axes[1].set_ylabel("Throttle")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Brake
    axes[2].plot(x, gt["brake"][:n], "b-", alpha=0.5, label="Expert", linewidth=1)
    axes[2].plot(x, bc["brake"][:n], "r-", alpha=0.7, label="BC", linewidth=1)
    axes[2].plot(x, rl["brake"][:n], "g-", alpha=0.7, label="RL", linewidth=1)
    axes[2].set_ylabel("Brake")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")
    bc_model = load_bc_model("checkpoints/best.pt", device)
    world_model = load_world_model("checkpoints/world_model_best.pt", device)
    rl_actor = load_rl_actor(
        "checkpoints/rl_actor_best.pt",
        world_model.rssm.full_state_dim,
        device,
    )

    # Load validation sequences
    val_dataset = SequenceDataset(
        data_root="data_raw/transfuser",
        towns=["Town05", "Town06"],
        seq_len=16,
        stride=16,  # No overlap for clean evaluation
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Evaluate
    print(f"\nEvaluating on {len(val_dataset)} validation sequences...")
    bc_results, rl_results, gt_results = evaluate_both(
        bc_model, world_model, rl_actor, val_loader, device, max_seqs=200
    )

    # Print comparison
    print_comparison(bc_results, rl_results, gt_results)

    # Plot
    plot_comparison(bc_results, rl_results, gt_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
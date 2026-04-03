"""
RL in Imagination — Phase 2, Step 2

This is the core of the project. We use the trained world model to
imagine driving trajectories, and train the policy with PPO to maximize
predicted rewards — all without CARLA, all on MPS.

Pipeline:
  1. Load trained world model (frozen) and BC policy (as starting point)
  2. Sample real frames from dataset, encode them into latent states
  3. From each latent state, imagine H steps forward using the policy
  4. World model predicts rewards for each imagined step
  5. Compute advantages and update policy with PPO
  6. Repeat

This follows the Dreamer approach: actor-critic learning in latent space.

Usage:
    python -m neurodriver.training.train_rl_imagination
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurodriver.data.sequence_dataset import SequenceDataset
from neurodriver.models.world_model import WorldModel
from neurodriver.utils.device import get_device


class ImagineActor(nn.Module):
    """
    Policy network that operates in the world model's latent space.

    Maps latent state [h, z] -> driving actions [steer, throttle, brake].

    Uses tanh-squashed Gaussian for exploration during training.
    """

    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: (B, state_dim) latent state from world model.
        Returns:
            action: (B, action_dim) sampled action.
            log_prob: (B, 1) log probability of the action.
            mean: (B, action_dim) mean action (for eval mode).
        """
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5, 2)
        std = log_std.exp()

        # Sample with reparameterization
        noise = torch.randn_like(mean)
        raw_action = mean + std * noise

        # Squash to valid ranges
        action = self._squash(raw_action)

        # Log probability (with squashing correction)
        log_prob = (
            -0.5 * ((raw_action - mean) / (std + 1e-8)).pow(2)
            - log_std
            - 0.5 * torch.log(torch.tensor(2.0 * 3.14159))
        ).sum(dim=-1, keepdim=True)

        # Squashing correction
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def _squash(self, raw: torch.Tensor) -> torch.Tensor:
        """Squash actions: steer to [-1,1], throttle/brake to [0,1]."""
        steer = torch.tanh(raw[:, 0:1])
        throttle = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get action for inference."""
        action, log_prob, mean = self.forward(state)
        if deterministic:
            return self._squash(mean)
        return action


class ImagineCritic(nn.Module):
    """Value function in latent space. Estimates expected return from a state."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def compute_lambda_returns(rewards, values, continues, gamma=0.99, lam=0.95):
    """
    Compute GAE-style lambda returns for imagined trajectories.

    Args:
        rewards: (B, H, 1) predicted rewards
        values: (B, H, 1) predicted values
        continues: (B, H, 1) continuation probabilities
        gamma: discount factor
        lam: GAE lambda
    Returns:
        returns: (B, H, 1) computed returns
    """
    B, H, _ = rewards.shape
    returns = torch.zeros_like(rewards)

    # Bootstrap from last value
    last_val = values[:, -1]

    for t in reversed(range(H)):
        if t == H - 1:
            next_val = last_val
        else:
            next_val = (1 - lam) * values[:, t + 1] + lam * returns[:, t + 1]

        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * next_val

    return returns


def imagine_and_learn(
    world_model: WorldModel,
    actor: ImagineActor,
    critic: ImagineCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    start_states: dict,
    horizon: int = 15,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_weight: float = 1e-3,
):
    """
    Core RL loop: imagine trajectories and update actor-critic.

    Args:
        world_model: Trained world model (frozen).
        actor: Policy to train.
        critic: Value function to train.
        start_states: Initial RSSM states from real data.
        horizon: Imagination horizon.
    Returns:
        Dict of training metrics.
    """
    # -- Imagine trajectory --
    state = {k: v.detach() for k, v in start_states.items()}

    imagined_states = []
    imagined_actions = []
    imagined_log_probs = []
    imagined_rewards = []
    imagined_continues = []

    for t in range(horizon):
        full_state = world_model.rssm.get_full_state(state)
        imagined_states.append(full_state)

        # Actor picks action
        action, log_prob, _ = actor(full_state)
        imagined_actions.append(action)
        imagined_log_probs.append(log_prob)

        # World model predicts next state
        state = world_model.rssm.imagine_step(state, action)

        # Predict reward and continuation
        next_full = world_model.rssm.get_full_state(state)
        reward = world_model.reward_model(next_full)
        cont = torch.sigmoid(world_model.continue_model(next_full))

        imagined_rewards.append(reward)
        imagined_continues.append(cont)

    # Stack: (B, H, dim)
    states_t = torch.stack(imagined_states, dim=1)
    rewards_t = torch.stack(imagined_rewards, dim=1)
    continues_t = torch.stack(imagined_continues, dim=1)
    log_probs_t = torch.stack(imagined_log_probs, dim=1)

    # -- Compute values and returns --
    with torch.no_grad():
        values_t = critic(states_t)
        returns_t = compute_lambda_returns(rewards_t, values_t, continues_t, gamma, lam)
        advantages = returns_t - values_t

    # -- Update critic --
    critic_optimizer.zero_grad()
    value_pred = critic(states_t.detach())
    critic_loss = F.mse_loss(value_pred, returns_t.detach())
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()

    # -- Update actor --
    actor_optimizer.zero_grad()

    # Recompute log probs (needed for gradient)
    new_actions, new_log_probs, _ = [], [], []
    for t in range(horizon):
        a, lp, _ = actor(states_t[:, t].detach())
        new_actions.append(a)
        new_log_probs.append(lp)
    new_log_probs_t = torch.stack(new_log_probs, dim=1)

    # Policy gradient with advantages
    actor_loss = -(new_log_probs_t * advantages.detach()).mean()

    # Entropy bonus for exploration
    entropy = -new_log_probs_t.mean()
    actor_loss = actor_loss - entropy_weight * entropy

    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "reward_mean": rewards_t.mean().item(),
        "reward_std": rewards_t.std().item(),
        "value_mean": values_t.mean().item(),
        "advantage_mean": advantages.mean().item(),
        "entropy": entropy.item(),
    }


def encode_initial_states(world_model, images, actions, device):
    """
    Encode real frames into RSSM latent states to start imagination from.

    Takes the first frame of each sequence, runs it through the encoder
    and one RSSM observe step to get a grounded initial state.
    """
    B = images.shape[0]

    # Use first frame and action
    first_img = images[:, 0].to(device)
    first_action = actions[:, 0].to(device)

    embed = world_model.encoder(first_img)
    state = world_model.rssm.initial_state(B, device)
    state = world_model.rssm.observe_step(state, first_action, embed)

    # Return only h and z (detached from encoder graph)
    return {"h": state["h"].detach(), "z": state["z"].detach()}


def train_rl(
    wm_checkpoint: str = "checkpoints/world_model_best.pt",
    data_root: str = "data_raw/transfuser",
    train_towns: list = None,
    horizon: int = 15,
    batch_size: int = 32,
    num_updates: int = 500,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    entropy_weight: float = 1e-3,
    gamma: float = 0.99,
    log_every: int = 20,
    checkpoint_dir: str = "checkpoints",
):
    device = get_device()
    print(f"Device: {device}")

    if train_towns is None:
        train_towns = ["Town01", "Town02", "Town03", "Town04"]

    # -- Load world model (frozen) --
    print(f"\nLoading world model from {wm_checkpoint}...")
    wm_ckpt = torch.load(wm_checkpoint, map_location=device, weights_only=False)

    world_model = WorldModel(
        stoch_dim=64, deter_dim=256, hidden_dim=256, action_dim=3, embed_dim=256
    ).to(device)
    world_model.load_state_dict(wm_ckpt["model_state_dict"])
    world_model.eval()

    # Freeze world model — we only train actor and critic
    for p in world_model.parameters():
        p.requires_grad = False

    print(f"  World model loaded (epoch {wm_ckpt['epoch']}, val_loss={wm_ckpt['val_loss']:.4f})")

    # -- Build actor-critic --
    state_dim = world_model.rssm.full_state_dim
    actor = ImagineActor(state_dim, action_dim=3, hidden_dim=256).to(device)
    critic = ImagineCritic(state_dim, hidden_dim=256).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Actor params: {actor_params:,}")
    print(f"  Critic params: {critic_params:,}")

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    # -- Data (for encoding initial states) --
    print(f"\nLoading data for initial state encoding...")
    dataset = SequenceDataset(
        data_root=data_root,
        towns=train_towns,
        seq_len=4,  # Only need a few frames to get initial state
        stride=8,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    # -- Training loop --
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting RL training: {num_updates} updates, horizon={horizon}\n")

    metrics_history = []
    best_reward = float("-inf")

    for update in range(1, num_updates + 1):
        t0 = time.time()

        # Get batch of real data for initial states
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # Encode initial states from real data
        with torch.no_grad():
            initial_states = encode_initial_states(
                world_model, batch["images"], batch["actions"], device
            )

        # Imagine and learn
        metrics = imagine_and_learn(
            world_model=world_model,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            start_states=initial_states,
            horizon=horizon,
            gamma=gamma,
            entropy_weight=entropy_weight,
        )

        elapsed = time.time() - t0
        metrics_history.append(metrics)

        # Logging
        if update % log_every == 0:
            # Average metrics over last log_every updates
            recent = metrics_history[-log_every:]
            avg = {k: sum(m[k] for m in recent) / len(recent) for k in recent[0]}

            print(
                f"Update {update:4d}/{num_updates} | "
                f"Reward: {avg['reward_mean']:+.4f} (std={avg['reward_std']:.3f}) | "
                f"Value: {avg['value_mean']:+.4f} | "
                f"Actor: {avg['actor_loss']:.4f} | "
                f"Critic: {avg['critic_loss']:.4f} | "
                f"Entropy: {avg['entropy']:.3f} | "
                f"{elapsed:.1f}s"
            )

            if avg["reward_mean"] > best_reward:
                best_reward = avg["reward_mean"]
                torch.save({
                    "update": update,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "reward_mean": best_reward,
                }, ckpt_dir / "rl_actor_best.pt")
                print(f"  New best actor saved (reward={best_reward:.4f})")

    # Save final
    torch.save({
        "update": num_updates,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "metrics_history": metrics_history,
    }, ckpt_dir / "rl_actor_final.pt")

    print(f"\nRL training complete. Best imagined reward: {best_reward:.4f}")
    print(f"Saved to: {ckpt_dir / 'rl_actor_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--wm-checkpoint", type=str, default="checkpoints/world_model_best.pt")
    args = parser.parse_args()

    train_rl(
        wm_checkpoint=args.wm_checkpoint,
        num_updates=args.num_updates,
        horizon=args.horizon,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_weight=args.entropy_weight,
    )
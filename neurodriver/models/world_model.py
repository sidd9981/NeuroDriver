"""
World Model for Autonomous Driving — RSSM (Recurrent State Space Model).

Based on DreamerV3 (Hafner et al., 2023) simplified for driving.

The world model learns to predict the future from actions alone,
without needing the simulator. This enables RL in "imagination":

  1. ENCODE: Real image -> compact latent state
  2. TRANSITION: (latent state, action) -> next latent state  
  3. DECODE: latent state -> predicted image features (for training)
  4. REWARD: latent state -> predicted reward (for RL)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              RSSM World Model                    │
    │                                                  │
    │  Image ──▶ Encoder ──▶ posterior z(t)            │
    │                          │                       │
    │  h(t-1), a(t-1) ──▶ GRU ──▶ h(t) ──▶ prior ẑ(t)│
    │                          │                       │
    │              [h(t), z(t)] = full state            │
    │                   │                              │
    │                   ├──▶ Decoder ──▶ reconstructed  │
    │                   ├──▶ Reward Model ──▶ reward    │
    │                   └──▶ Continue Model ──▶ done?   │
    └─────────────────────────────────────────────────┘

The key insight: during training, we use the ENCODER (posterior) to get
accurate states from real images. During imagination, we use only the
TRANSITION model (prior) to predict states from actions — no images needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class ImageEncoder(nn.Module):
    """
    Encode an image into a compact feature vector.
    
    Uses a small CNN (not the full ResNet — we want this to be fast
    since it runs on every frame during world model training).
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        # Simple CNN encoder: 3x256x256 -> feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                # 4x4
        )
        self.fc = nn.Linear(256 * 4 * 4, feature_dim)
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor.
        Returns:
            (B, feature_dim) feature vector.
        """
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class RSSM(nn.Module):
    """
    Recurrent State Space Model — the core of the world model.
    
    State has two parts:
      - Deterministic state h: captures long-term memory (GRU hidden state)
      - Stochastic state z: captures uncertainty about the current state
    
    Two modes:
      - OBSERVE: given real image, compute posterior z (accurate)
      - IMAGINE: given only previous state + action, compute prior z (predicted)
    
    Args:
        stoch_dim: Dimension of stochastic state z
        deter_dim: Dimension of deterministic state h (GRU hidden)
        hidden_dim: Dimension of MLP hidden layers
        action_dim: Dimension of action space (3: steer, throttle, brake)
        embed_dim: Dimension of image embedding from encoder
    """
    
    def __init__(
        self,
        stoch_dim: int = 64,
        deter_dim: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        
        # Deterministic transition: (h, z, a) -> next h
        self.gru_input = nn.Linear(stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        
        # Prior: h -> predicted z (used during imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),  # mean + log_std
        )
        
        # Posterior: (h, image_embed) -> accurate z (used during training)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),  # mean + log_std
        )
    
    def initial_state(self, batch_size: int, device: torch.device):
        """Create zero-initialized state for the start of a sequence."""
        return {
            "h": torch.zeros(batch_size, self.deter_dim, device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }
    
    def observe_step(
        self,
        prev_state: dict,
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> dict:
        """
        One step with observation (training mode).
        
        Uses the image embedding to compute an accurate posterior state.
        
        Args:
            prev_state: dict with 'h' (B, deter_dim) and 'z' (B, stoch_dim)
            action: (B, action_dim) action taken
            embed: (B, embed_dim) image embedding from encoder
        
        Returns:
            dict with h, z, prior_mean, prior_std, post_mean, post_std
        """
        # Deterministic transition
        gru_in = self.gru_input(torch.cat([prev_state["z"], action], dim=-1))
        gru_in = F.elu(gru_in)
        h = self.gru(gru_in, prev_state["h"])
        
        # Prior (what the model predicts without seeing the image)
        prior_params = self.prior_net(h)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1  # Min std to prevent collapse
        
        # Posterior (accurate state using the actual image)
        post_params = self.posterior_net(torch.cat([h, embed], dim=-1))
        post_mean, post_log_std = post_params.chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1
        
        # Sample z from posterior (with reparameterization for gradients)
        z = post_mean + post_std * torch.randn_like(post_std)
        
        return {
            "h": h,
            "z": z,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "post_mean": post_mean,
            "post_std": post_std,
        }
    
    def imagine_step(
        self,
        prev_state: dict,
        action: torch.Tensor,
    ) -> dict:
        """
        One step WITHOUT observation (imagination mode for RL).
        
        Uses only the prior — no image needed.
        
        Args:
            prev_state: dict with 'h' and 'z'
            action: (B, action_dim) action to take
        
        Returns:
            dict with h, z (sampled from prior)
        """
        # Deterministic transition
        gru_in = self.gru_input(torch.cat([prev_state["z"], action], dim=-1))
        gru_in = F.elu(gru_in)
        h = self.gru(gru_in, prev_state["h"])
        
        # Prior only (no image to compute posterior)
        prior_params = self.prior_net(h)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1
        
        z = prior_mean + prior_std * torch.randn_like(prior_std)
        
        return {"h": h, "z": z}
    
    def get_full_state(self, state: dict) -> torch.Tensor:
        """Concatenate h and z into a single state vector."""
        return torch.cat([state["h"], state["z"]], dim=-1)
    
    @property
    def full_state_dim(self) -> int:
        return self.deter_dim + self.stoch_dim


class RewardModel(nn.Module):
    """
    Predict reward from the world model's latent state.
    
    During world model training, we train this to predict a driving reward
    computed from the ground truth data (speed, lane centering, collisions).
    
    During RL imagination, this provides the reward signal.
    """
    
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
        """
        Args:
            state: (B, state_dim) concatenated [h, z] from RSSM.
        Returns:
            (B, 1) predicted reward.
        """
        return self.net(state)


class ContinueModel(nn.Module):
    """
    Predict whether the episode continues (not done) from latent state.
    
    Outputs probability of continuation. Used during imagination to know
    when to stop a trajectory (e.g., after a collision).
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns logits (apply sigmoid for probability)."""
        return self.net(state)


class WorldModel(nn.Module):
    """
    Complete World Model assembling all components.
    
    Training:
        1. Encode sequence of real images
        2. Run RSSM with observations to get posterior states
        3. Decode states to reconstruct features
        4. Predict rewards from states
        5. Loss = reconstruction + reward_prediction + KL(posterior || prior)
    
    Imagination (for RL):
        1. Start from a real encoded state
        2. Use imagine_step repeatedly with policy actions
        3. Get predicted rewards from reward model
        4. No images needed — pure tensor operations on MPS
    
    Args:
        stoch_dim: Stochastic state dimension
        deter_dim: Deterministic (GRU) state dimension
        hidden_dim: MLP hidden dimension
        action_dim: Action space dimension
        embed_dim: Image encoder output dimension
    """
    
    def __init__(
        self,
        stoch_dim: int = 64,
        deter_dim: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        
        self.encoder = ImageEncoder(embed_dim)
        self.rssm = RSSM(stoch_dim, deter_dim, hidden_dim, action_dim, embed_dim)
        
        state_dim = deter_dim + stoch_dim
        self.reward_model = RewardModel(state_dim, hidden_dim)
        self.continue_model = ContinueModel(state_dim, hidden_dim)
        
        # Decoder: reconstruct image embedding from state (for training)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def observe_sequence(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Process a sequence of real (image, action) pairs.
        
        Args:
            images: (B, T, 3, H, W) sequence of images
            actions: (B, T, action_dim) sequence of actions
        
        Returns:
            Dict with states, priors, posteriors, decoded features for each step.
        """
        B, T = images.shape[:2]
        device = images.device
        
        # Encode all images at once for efficiency
        flat_images = images.reshape(B * T, *images.shape[2:])
        flat_embeds = self.encoder(flat_images)
        embeds = flat_embeds.reshape(B, T, -1)  # (B, T, embed_dim)
        
        # Roll out RSSM with observations
        state = self.rssm.initial_state(B, device)
        
        all_states = []
        all_priors = []
        all_posteriors = []
        
        for t in range(T):
            # Use previous action (zero for first step)
            action = actions[:, t]
            
            state = self.rssm.observe_step(state, action, embeds[:, t])
            
            full_state = self.rssm.get_full_state(state)
            all_states.append(full_state)
            all_priors.append((state["prior_mean"], state["prior_std"]))
            all_posteriors.append((state["post_mean"], state["post_std"]))
        
        # Stack over time
        states = torch.stack(all_states, dim=1)  # (B, T, state_dim)
        
        # Decode states back to embeddings
        decoded = self.decoder(states)  # (B, T, embed_dim)
        
        # Predict rewards
        rewards = self.reward_model(states)  # (B, T, 1)
        
        # Predict continuation
        continues = self.continue_model(states)  # (B, T, 1)
        
        return {
            "states": states,
            "embeds": embeds,
            "decoded": decoded,
            "rewards": rewards,
            "continues": continues,
            "priors": all_priors,
            "posteriors": all_posteriors,
        }
    
    def imagine_trajectory(
        self,
        initial_state: dict,
        policy: nn.Module,
        horizon: int = 15,
    ) -> dict:
        """
        Imagine a trajectory using the policy, WITHOUT real images.
        
        This is where RL training happens — entirely in latent space.
        
        Args:
            initial_state: dict with 'h' and 'z' from a real observation
            policy: Actor network that maps state -> action
            horizon: How many steps to imagine
        
        Returns:
            Dict with imagined states, actions, rewards, continues.
        """
        state = initial_state
        
        all_states = []
        all_actions = []
        all_rewards = []
        all_continues = []
        
        for t in range(horizon):
            full_state = self.rssm.get_full_state(state)
            
            # Policy decides action
            action = policy(full_state)
            
            # World model predicts next state
            state = self.rssm.imagine_step(state, action)
            
            # Predict reward and continuation
            next_full = self.rssm.get_full_state(state)
            reward = self.reward_model(next_full)
            cont = torch.sigmoid(self.continue_model(next_full))
            
            all_states.append(next_full)
            all_actions.append(action)
            all_rewards.append(reward)
            all_continues.append(cont)
        
        return {
            "states": torch.stack(all_states, dim=1),      # (B, H, state_dim)
            "actions": torch.stack(all_actions, dim=1),     # (B, H, action_dim)
            "rewards": torch.stack(all_rewards, dim=1),     # (B, H, 1)
            "continues": torch.stack(all_continues, dim=1), # (B, H, 1)
        }


def world_model_loss(
    output: dict,
    target_rewards: torch.Tensor,
    kl_weight: float = 1.0,
    kl_balance: float = 0.8,
) -> dict:
    """
    Compute world model training loss.
    
    Components:
        1. Reconstruction: decoded features should match encoded features
        2. Reward: predicted rewards should match computed rewards
        3. KL divergence: posterior should be close to prior
           (teaches the prior to be accurate for imagination)
    
    Args:
        output: Dict from observe_sequence
        target_rewards: (B, T, 1) ground truth rewards
        kl_weight: Weight on KL loss
        kl_balance: Balance between training prior vs posterior
    
    Returns:
        Dict with loss components.
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(output["decoded"], output["embeds"].detach())
    
    # Reward prediction loss
    reward_loss = F.mse_loss(output["rewards"], target_rewards)
    
    # KL divergence between posterior and prior
    kl_total = torch.tensor(0.0, device=output["states"].device)
    T = len(output["priors"])
    
    for t in range(T):
        prior_mean, prior_std = output["priors"][t]
        post_mean, post_std = output["posteriors"][t]
        
        prior_dist = Normal(prior_mean, prior_std)
        post_dist = Normal(post_mean, post_std)
        
        # KL(posterior || prior) — teaches prior to match posterior
        kl = kl_divergence(post_dist, prior_dist).sum(dim=-1).mean()
        kl_total = kl_total + kl
    
    kl_loss = kl_total / max(T, 1)
    
    # Total loss
    total = recon_loss + reward_loss + kl_weight * kl_loss
    
    return {
        "total": total,
        "recon": recon_loss.detach(),
        "reward": reward_loss.detach(),
        "kl": kl_loss.detach(),
    }


if __name__ == "__main__":
    """Smoke test: verify shapes, gradients, imagination."""
    from neurodriver.utils.device import get_device
    
    device = get_device()
    print(f"Testing WorldModel on: {device}")
    
    # Build world model
    wm = WorldModel(
        stoch_dim=64,
        deter_dim=256,
        hidden_dim=256,
        action_dim=3,
        embed_dim=256,
    ).to(device)
    
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"World model parameters: {n_params:,}")
    
    # Test observe_sequence
    B, T = 4, 10
    images = torch.randn(B, T, 3, 256, 256, device=device)
    actions = torch.randn(B, T, 3, device=device)
    
    print(f"\nObserve sequence: B={B}, T={T}")
    output = wm.observe_sequence(images, actions)
    print(f"  States: {output['states'].shape}")
    print(f"  Decoded: {output['decoded'].shape}")
    print(f"  Rewards: {output['rewards'].shape}")
    
    # Test loss
    target_rewards = torch.randn(B, T, 1, device=device)
    losses = world_model_loss(output, target_rewards)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test gradient flow
    losses["total"].backward()
    no_grad = [n for n, p in wm.named_parameters() if p.grad is None]
    if no_grad:
        print(f"\n No gradient: {no_grad[:5]}...")
    else:
        print(f"\nGradient flow: PASSED ")
    
    # Test imagination
    wm.zero_grad()
    print(f"\nImagination test:")
    
    # Get initial state from a real observation
    state = wm.rssm.initial_state(B, device)
    embed = wm.encoder(images[:, 0])
    state = wm.rssm.observe_step(state, actions[:, 0], embed)
    
    # Simple dummy policy for testing
    class DummyPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Linear(state_dim, action_dim)
        def forward(self, state):
            return torch.tanh(self.net(state))
    
    policy = DummyPolicy(wm.rssm.full_state_dim, 3).to(device)
    
    imagined = wm.imagine_trajectory(state, policy, horizon=15)
    print(f"  Imagined states: {imagined['states'].shape}")
    print(f"  Imagined actions: {imagined['actions'].shape}")
    print(f"  Imagined rewards: {imagined['rewards'].shape}")
    
    # Verify we can backprop through imagination
    imag_return = imagined["rewards"].sum()
    imag_return.backward()
    print(f"  Backprop through imagination: PASSED ")
    
    print(f"\n WorldModel test PASSED on {device}!")
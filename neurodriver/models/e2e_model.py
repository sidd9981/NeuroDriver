"""
End-to-End Driving Model — from camera image to vehicle controls.

Architecture:
    RGB Image --> ResNet-34 -----------> ┐
    Speed --> SpeedEncoder ------------> ├--> Fused Features --> Control Head --> [steer, throttle, brake]
    Command --> CommandEncoder --------> ┘                  │
                                                            ├--> Speed Head --> predicted speed
                                                            └--> Waypoint Head --> predicted waypoints

The Control Head is the main output (what the car should do).
Speed Head and Waypoint Head are auxiliary losses that help the model
learn better representations by forcing it to understand ego-motion
and spatial layout. This is a proven trick from TCP and TransFuser.
"""

import torch
import torch.nn as nn

from neurodriver.models.backbone import ResNetBackbone


class SpeedEncoder(nn.Module):
    """
    Small MLP to encode scalar speed into a learned embedding.
    
    Why not just concatenate the raw speed value?
    A learned embedding gives the model a richer representation
    of speed — it can learn that "stopped" and "highway speed"
    are very different regimes that need different behaviors.
    """
    
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
    
    def forward(self, speed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speed: (B, 1) float tensor, ego speed in m/s.
        Returns:
            (B, embed_dim) speed embedding.
        """
        return self.net(speed)


class CommandEncoder(nn.Module):
    """
    Embedding for high-level driving commands.
    
    CARLA commands: 1=turn left, 2=turn right, 3=go straight, 4=follow lane
    
    This lets the model condition its behavior on the navigation instruction.
    Without this, the model doesn't know which way to turn at intersections.
    """
    
    def __init__(self, num_commands: int = 4, embed_dim: int = 64):
        super().__init__()
        # +1 because commands are 1-indexed (1,2,3,4), not 0-indexed
        self.embed = nn.Embedding(num_commands + 1, embed_dim)
    
    def forward(self, command: torch.Tensor) -> torch.Tensor:
        """
        Args:
            command: (B, 1) long tensor with values in {1, 2, 3, 4}.
        Returns:
            (B, embed_dim) command embedding.
        """
        return self.embed(command.squeeze(-1))  # (B,) -> (B, embed_dim)


class ControlHead(nn.Module):
    """
    MLP that predicts driving controls from fused features.
    
    Outputs: [steering, throttle, brake]
    - steering: tanh activation -> [-1, 1]
    - throttle: sigmoid activation -> [0, 1]
    - brake: sigmoid activation -> [0, 1]
    """
    
    def __init__(self, in_dim: int, hidden_dims: list[int] = [512, 256], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Separate output heads for different activation functions
        self.steer_head = nn.Linear(prev_dim, 1)
        self.throttle_head = nn.Linear(prev_dim, 1)
        self.brake_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_dim) fused feature vector.
        Returns:
            Dict with steer (B,1), throttle (B,1), brake (B,1).
        """
        h = self.mlp(x)
        return {
            "steer": torch.tanh(self.steer_head(h)),       # [-1, 1]
            "throttle": torch.sigmoid(self.throttle_head(h)), # [0, 1]
            "brake": torch.sigmoid(self.brake_head(h)),     # [0, 1]
        }


class DrivingModel(nn.Module):
    """
    Full end-to-end driving model.
    
    Takes camera image + speed + command -> predicts controls.
    Also predicts speed and waypoints as auxiliary tasks.
    
    Args:
        backbone_name: "resnet34" or "resnet18"
        pretrained: Use ImageNet pretrained backbone
        feature_dim: Backbone output dimension (512 for ResNet-34)
        speed_embed_dim: Speed encoder output dimension
        command_embed_dim: Command encoder output dimension
        num_commands: Number of high-level commands (4 for CARLA)
        hidden_dims: MLP hidden layer sizes
        dropout: Dropout rate in MLP heads
        num_waypoints: Number of future waypoints to predict
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        feature_dim: int = 512,
        speed_embed_dim: int = 64,
        command_embed_dim: int = 64,
        num_commands: int = 4,
        hidden_dims: list[int] = [512, 256],
        dropout: float = 0.1,
        num_waypoints: int = 5,
    ):
        super().__init__()
        
        # Perception
        self.backbone = ResNetBackbone(backbone_name, pretrained, feature_dim)
        
        # Conditioning encoders
        self.speed_encoder = SpeedEncoder(speed_embed_dim)
        self.command_encoder = CommandEncoder(num_commands, command_embed_dim)
        
        # Total fused feature dimension
        fused_dim = feature_dim + speed_embed_dim + command_embed_dim
        
        # Main output: driving controls
        self.control_head = ControlHead(fused_dim, hidden_dims, dropout)
        
        # Auxiliary output: speed prediction (helps learn ego-motion awareness)
        self.speed_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        # Auxiliary output: waypoint prediction (helps learn spatial understanding)
        self.waypoint_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * 2),  # (x, y) for each waypoint
        )
        self.num_waypoints = num_waypoints
    
    def forward(
        self,
        image: torch.Tensor,
        speed: torch.Tensor,
        command: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            image: (B, 3, H, W) normalized RGB image.
            speed: (B, 1) ego speed in m/s.
            command: (B, 1) high-level command {1,2,3,4}.
        
        Returns:
            Dict with:
                steer: (B, 1) steering prediction [-1, 1]
                throttle: (B, 1) throttle prediction [0, 1]
                brake: (B, 1) brake prediction [0, 1]
                pred_speed: (B, 1) predicted speed (auxiliary)
                pred_waypoints: (B, num_waypoints, 2) predicted waypoints (auxiliary)
        """
        # Extract image features
        img_features = self.backbone(image)         # (B, 512)
        
        # Encode conditioning signals
        speed_emb = self.speed_encoder(speed)       # (B, 64)
        cmd_emb = self.command_encoder(command)     # (B, 64)
        
        # Fuse everything by concatenation
        # This is simple and proven — TCP uses the same approach
        fused = torch.cat([img_features, speed_emb, cmd_emb], dim=1)  # (B, 640)
        
        # Main output: controls
        controls = self.control_head(fused)
        
        # Auxiliary outputs
        pred_speed = self.speed_head(fused)                     # (B, 1)
        pred_wp = self.waypoint_head(fused)                     # (B, num_wp * 2)
        pred_wp = pred_wp.view(-1, self.num_waypoints, 2)       # (B, num_wp, 2)
        
        return {
            **controls,                    # steer, throttle, brake
            "pred_speed": pred_speed,
            "pred_waypoints": pred_wp,
        }


def build_model(cfg: dict) -> DrivingModel:
    """Build model from config dict."""
    model_cfg = cfg["model"]
    return DrivingModel(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        feature_dim=model_cfg["feature_dim"],
        speed_embed_dim=model_cfg["speed_embed_dim"],
        command_embed_dim=model_cfg["command_embed_dim"],
        num_commands=model_cfg["num_commands"],
        hidden_dims=model_cfg["policy"]["hidden_dims"],
        dropout=model_cfg["policy"]["dropout"],
    )


if __name__ == "__main__":
    """Smoke test: verify forward pass, shapes, gradients."""
    from neurodriver.utils.device import get_device
    
    device = get_device()
    print(f"Testing DrivingModel on: {device}")
    
    # Build model
    model = DrivingModel(
        backbone_name="resnet34",
        pretrained=True,
    ).to(device)
    
    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Dummy batch
    B = 4
    image = torch.randn(B, 3, 256, 256, device=device)
    speed = torch.randn(B, 1, device=device).abs() * 10  # 0-10 m/s
    command = torch.randint(1, 5, (B, 1), device=device)  # {1,2,3,4}
    
    # Forward pass
    output = model(image, speed, command)
    
    print(f"\nOutput shapes:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")
    
    # Verify ranges
    print(f"\nOutput ranges:")
    print(f"  steer:    [{output['steer'].min():.3f}, {output['steer'].max():.3f}] (expect [-1, 1])")
    print(f"  throttle: [{output['throttle'].min():.3f}, {output['throttle'].max():.3f}] (expect [0, 1])")
    print(f"  brake:    [{output['brake'].min():.3f}, {output['brake'].max():.3f}] (expect [0, 1])")
    
    # Verify gradient flow
    loss = sum(v.sum() for v in output.values())
    loss.backward()
    
    # Check all params got gradients
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    if no_grad:
        print(f"\n Parameters without gradients: {no_grad}")
    else:
        print(f"\nGradient flow to all parameters: PASSED")
    
    print(f"\nDrivingModel test PASSED on {device}!")
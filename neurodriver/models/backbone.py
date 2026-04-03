"""
Image backbone for driving — ResNet-34 with ImageNet pretraining.

The backbone takes a (B, 3, H, W) image tensor and outputs
a (B, 512) feature vector.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    """
    ResNet image encoder that outputs a flat feature vector.
    
    Removes the final classification head (fc layer) from ResNet
    and replaces it with adaptive average pooling -> flat features.
    
    Args:
        name: Which ResNet variant. "resnet34" or "resnet18".
        pretrained: Use ImageNet pretrained weights.
        feature_dim: Output feature dimension (512 for ResNet-34/18).
    """
    
    def __init__(
        self,
        name: str = "resnet34",
        pretrained: bool = True,
        feature_dim: int = 512,
    ):
        super().__init__()
        
        # Load pretrained ResNet
        if name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
        elif name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {name}. Use 'resnet34' or 'resnet18'.")
        
        # Take everything except the final fc layer
        # ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # We keep everything up to and including avgpool
        self.features = nn.Sequential(
            resnet.conv1,    # (B, 64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B, 64, H/4, W/4)
            resnet.layer1,   # (B, 64, H/4, W/4)
            resnet.layer2,   # (B, 128, H/8, W/8)
            resnet.layer3,   # (B, 256, H/16, W/16)
            resnet.layer4,   # (B, 512, H/32, W/32)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 512, 1, 1)
        self.feature_dim = feature_dim
        
        # Verify output dimension matches expected
        # ResNet-18 and ResNet-34 both output 512
        actual_dim = resnet.fc.in_features
        assert actual_dim == feature_dim, (
            f"ResNet {name} outputs {actual_dim}-dim features, "
            f"but feature_dim={feature_dim} was specified."
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor, normalized with ImageNet stats.
        
        Returns:
            (B, feature_dim) feature vector.
        """
        x = self.features(x)   # (B, 512, H/32, W/32)
        x = self.pool(x)       # (B, 512, 1, 1)
        x = x.flatten(1)       # (B, 512)
        return x


if __name__ == "__main__":
    """Smoke test: verify shapes and that MPS works."""
    from neurodriver.utils.device import get_device
    
    device = get_device()
    print(f"Testing on device: {device}")
    
    # Create backbone
    backbone = ResNetBackbone(name="resnet34", pretrained=True).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in backbone.parameters())
    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 256, 256, device=device)
    features = backbone(dummy_input)
    print(f"Input:  {dummy_input.shape}")
    print(f"Output: {features.shape}")
    
    # Verify gradient flow
    loss = features.sum()
    loss.backward()
    print(f"Gradient flow: PASSED")
    
    print(f"\nBackbone test PASSED on {device}!")
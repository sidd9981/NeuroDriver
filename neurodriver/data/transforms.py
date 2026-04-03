"""
Image transforms for training and validation.

Key design decisions:
- No horizontal flip: Flipping reverses left/right steering semantics.
  A left turn becomes a right turn, which corrupts the labels.
- Color jitter: Simulates weather/lighting changes (sun glare, overcast, etc.)
- Normalize with ImageNet stats: Because we use ImageNet-pretrained ResNet.
"""

from torchvision import transforms


# ImageNet normalization — required when using pretrained ResNet/ViT
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: tuple[int, int] = (256, 256)) -> transforms.Compose:
    """
    Training augmentations. Designed for driving images.
    
    Args:
        image_size: (H, W) output size.
    
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        
        # Color augmentation — simulates weather/lighting variation
        # These values are from the TCP paper's augmentation pipeline
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,  # Small hue shift — don't want to turn sky green
        ),
        
        # Convert to tensor: HWC uint8 [0,255] -> CHW float [0,1]
        transforms.ToTensor(),
        
        # Normalize with ImageNet stats for pretrained backbone
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: tuple[int, int] = (256, 256)) -> transforms.Compose:
    """
    Validation transforms. No augmentation — just resize and normalize.
    
    Args:
        image_size: (H, W) output size.
    
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.
    
    Args:
        tensor: (3, H, W) or (B, 3, H, W) normalized tensor.
    
    Returns:
        Tensor with pixel values back in [0, 1] range.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return (tensor * std.to(tensor.device) + mean.to(tensor.device)).clamp(0, 1)


# Need torch for denormalize
import torch


if __name__ == "__main__":
    """Quick visual test of transforms."""
    import numpy as np
    from PIL import Image
    
    # Create a dummy driving image
    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(dummy)
    
    # Test train transforms
    train_tf = get_train_transforms((256, 256))
    tensor = train_tf(img)
    print(f"Train transform output: shape={tensor.shape}, min={tensor.min():.3f}, max={tensor.max():.3f}")
    
    # Test val transforms
    val_tf = get_val_transforms((256, 256))
    tensor = val_tf(img)
    print(f"Val transform output:   shape={tensor.shape}, min={tensor.min():.3f}, max={tensor.max():.3f}")
    
    # Test denormalization
    recovered = denormalize(tensor)
    print(f"Denormalized:           shape={recovered.shape}, min={recovered.min():.3f}, max={recovered.max():.3f}")
    
    print("\nTransforms test PASSED!")
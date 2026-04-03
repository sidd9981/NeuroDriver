"""
Device selection utility for Mac M3 Pro (MPS), CUDA, or CPU fallback.

Usage:
    from neurodriver.utils.device import get_device
    device = get_device()  # Returns best available device
    
Run standalone to verify your setup:
    python -m neurodriver.utils.device
"""

import torch


def get_device(force: str | None = None) -> torch.device:
    """
    Get the best available compute device.
    
    Priority: forced choice > MPS (Apple Silicon) > CUDA > CPU
    
    Args:
        force: Override auto-detection. One of "mps", "cuda", "cpu", or None.
    
    Returns:
        torch.device for tensor allocation and model placement.
    """
    if force is not None:
        return torch.device(force)
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def device_info() -> dict:
    """Return a dictionary of device capabilities for logging."""
    info = {
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(get_device()),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
    
    return info


if __name__ == "__main__":
    # Quick verification when run directly
    print("=" * 50)
    print("  NeuroDriver — Device Check")
    print("=" * 50)
    
    info = device_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    device = get_device()
    print(f"\n  Using device: {device}")
    
    # Quick smoke test
    print("\n  Running smoke test...")
    x = torch.randn(64, 512, device=device)
    y = torch.nn.functional.relu(x @ x.T)
    print(f"  Matrix multiply on {device}: shape={y.shape}, dtype={y.dtype}")
    print(f"  Smoke test PASSED ✓")
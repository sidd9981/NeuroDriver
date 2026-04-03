"""
Export trained BC model to TorchScript and ONNX for C++ inference.

TorchScript: used by our C++ inference bridge (LibTorch)
ONNX:        portable format, can be used with ONNX Runtime, TensorRT, etc.

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --checkpoint checkpoints/best.pt --output-dir cpp/models
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from neurodriver.models.e2e_model import DrivingModel
from neurodriver.utils.device import get_device


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
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
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model, cfg


class DrivingModelWrapper(torch.nn.Module):
    """
    Wrapper that bundles ImageNet normalization into the model.

    The C++ side sends raw [0,1] float images. This wrapper applies
    ImageNet normalization internally so the C++ code doesn't have to
    know about mean/std values.

    Also flattens the output dict into a single tensor [steer, throttle, brake, pred_speed]
    since TorchScript trace doesn't support dict outputs cleanly.
    """

    def __init__(self, model: DrivingModel):
        super().__init__()
        self.model = model

        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(
        self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image:   (B, 3, 256, 256) float tensor in [0, 1] range
            speed:   (B, 1) float tensor, m/s
            command: (B, 1) long tensor, {1, 2, 3, 4}

        Returns:
            (B, 4) float tensor: [steer, throttle, brake, pred_speed]
        """
        # Normalize
        image = (image - self.mean) / self.std

        # Forward
        out = self.model(image, speed, command)

        # Pack into single tensor for clean TorchScript output
        return torch.cat([
            out["steer"],       # (B, 1)
            out["throttle"],    # (B, 1)
            out["brake"],       # (B, 1)
            out["pred_speed"],  # (B, 1)
        ], dim=1)  # (B, 4)


def verify_outputs(model_wrapped, ts_model, device):
    """Verify TorchScript model produces same outputs as Python model."""
    image = torch.rand(1, 3, 256, 256, device=device)
    speed = torch.tensor([[7.0]], device=device)
    command = torch.tensor([[4]], device=device)

    with torch.no_grad():
        py_out = model_wrapped(image, speed, command)
        ts_out = ts_model(image, speed, command)

    diff = (py_out - ts_out).abs().max().item()
    print(f"  Max output difference: {diff:.8f}")
    assert diff < 1e-5, f"Output mismatch! Max diff: {diff}"
    print(f"  Verification PASSED")

    # Print sample outputs
    print(f"  Sample output: steer={ts_out[0,0]:.4f}, "
          f"throttle={ts_out[0,1]:.4f}, brake={ts_out[0,2]:.4f}, "
          f"speed={ts_out[0,3]:.4f}")


def export_torchscript(model_wrapped, output_path, device):
    """Export to TorchScript via tracing."""
    print("\nExporting TorchScript...")

    # Example inputs for tracing
    example_image = torch.rand(1, 3, 256, 256, device=device)
    example_speed = torch.tensor([[7.0]], device=device)
    example_command = torch.tensor([[4]], device=device)

    with torch.no_grad():
        traced = torch.jit.trace(
            model_wrapped,
            (example_image, example_speed, example_command),
        )

    # Save
    traced.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Reload and verify
    loaded = torch.jit.load(str(output_path), map_location=device)
    loaded.eval()
    verify_outputs(model_wrapped, loaded, device)

    return traced


def export_onnx(model_wrapped, output_path, device):
    """Export to ONNX format."""
    print("\nExporting ONNX...")

    example_image = torch.rand(1, 3, 256, 256, device=device)
    example_speed = torch.tensor([[7.0]], device=device)
    example_command = torch.tensor([[4]], device=device)

    torch.onnx.export(
        model_wrapped,
        (example_image, example_speed, example_command),
        str(output_path),
        input_names=["image", "speed", "command"],
        output_names=["controls"],
        dynamic_axes={
            "image": {0: "batch"},
            "speed": {0: "batch"},
            "command": {0: "batch"},
            "controls": {0: "batch"},
        },
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Verify with ONNX Runtime if available
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_path))
        onnx_out = sess.run(None, {
            "image": example_image.cpu().numpy(),
            "speed": example_speed.cpu().numpy(),
            "command": example_command.cpu().numpy().astype(np.int64),
        })

        with torch.no_grad():
            py_out = model_wrapped(example_image, example_speed, example_command)

        diff = abs(py_out.cpu().numpy() - onnx_out[0]).max()
        print(f"  ONNX Runtime verification: max diff = {diff:.8f}")
        assert diff < 1e-4, f"ONNX output mismatch! Max diff: {diff}"
        print(f"  ONNX verification PASSED")
    except ImportError:
        print(f"  onnxruntime not installed — skipping ONNX verification")
        print(f"  Install with: pip install onnxruntime")


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript + ONNX")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="cpp/models",
        help="Output directory for exported models",
    )
    args = parser.parse_args()

    # Use CPU for export (most portable)
    device = torch.device("cpu")
    print(f"Export device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint, device)

    # Wrap with normalization
    wrapped = DrivingModelWrapper(model).to(device)
    wrapped.eval()

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export TorchScript
    ts_path = out_dir / "driving_model.pt"
    export_torchscript(wrapped, ts_path, device)

    # Export ONNX
    onnx_path = out_dir / "driving_model.onnx"
    export_onnx(wrapped, onnx_path, device)

    # Summary
    print("\n")
    print("  Export Complete!")
    print("\n")
    print(f"  TorchScript: {ts_path}")
    print(f"  ONNX:        {onnx_path}")
    print(f"\n  Next: cd cpp && mkdir build && cd build")
    print(f"        cmake .. && make")
    print(f"        ./neurodriver_inference ../models/driving_model.pt")


if __name__ == "__main__":
    main()
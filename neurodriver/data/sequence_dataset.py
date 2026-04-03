"""
Sequence Dataset for World Model Training.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from neurodriver.data.transforms import get_val_transforms
from neurodriver.data.reward import compute_reward_v2


class SequenceDataset(Dataset):
    """
    Returns sequences of consecutive (image, action, reward) tuples
    from the same route, for world model training.

    Args:
        data_root: Path to dataset root (TransFuser format).
        towns: List of town names to include.
        seq_len: Number of consecutive frames per sequence.
        image_size: (H, W) for resized images.
        stride: Step between sequence start indices within a route.
    """

    def __init__(
        self,
        data_root: str,
        towns: list[str] = None,
        seq_len: int = 16,
        image_size: tuple[int, int] = (256, 256),
        stride: int = 4,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.transform = get_val_transforms(image_size)

        self.sequences = []
        self._build_sequences(towns, stride)

        print(
            f"SequenceDataset: {len(self.sequences)} sequences "
            f"(seq_len={seq_len}, stride={stride})"
        )

    def _build_sequences(self, towns, stride):
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

        for route_dir in sorted(self.data_root.iterdir()):
            if not route_dir.is_dir():
                continue

            if towns is not None:
                route_lower = route_dir.name.lower()
                if not any(t.lower() in route_lower for t in towns):
                    continue

            rgb_dir = route_dir / "rgb"
            meas_dir = route_dir / "measurements"

            if not rgb_dir.exists() or not meas_dir.exists():
                continue

            rgb_files = {}
            for ext in ("*.png", "*.jpg"):
                for f in rgb_dir.glob(ext):
                    rgb_files[f.stem] = str(f)

            meas_files = {f.stem: str(f) for f in meas_dir.glob("*.json")}
            valid_frames = sorted(rgb_files.keys() & meas_files.keys())

            if len(valid_frames) < self.seq_len:
                continue

            frames = []
            for fid in valid_frames:
                frames.append({
                    "rgb_path": rgb_files[fid],
                    "meas_path": meas_files[fid],
                })

            for start in range(0, len(frames) - self.seq_len + 1, stride):
                self.sequences.append(frames[start : start + self.seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            images: (T, 3, H, W) float tensor
            actions: (T, 3) float tensor [steer, throttle, brake]
            rewards: (T, 1) float tensor 
            speeds: (T, 1) float tensor
        """
        seq = self.sequences[idx]

        images = []
        actions = []
        rewards = []
        speeds = []
        prev_meas = None

        for frame in seq:
            img = Image.open(frame["rgb_path"]).convert("RGB")
            img = self.transform(img)
            images.append(img)

            with open(frame["meas_path"], "r") as f:
                meas = json.load(f)

            steer = float(meas.get("steer", 0.0))
            throttle = float(meas.get("throttle", 0.0))
            brake = float(meas.get("brake", 0.0))
            speed = float(meas.get("speed", 0.0))

            actions.append([steer, throttle, brake])
            speeds.append([speed])
            rewards.append([compute_reward_v2(meas, prev_meas)])

            prev_meas = meas

        return {
            "images": torch.stack(images),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "speeds": torch.tensor(speeds, dtype=torch.float32),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = SequenceDataset(
        data_root="data_raw/transfuser",
        towns=["Town01", "Town02", "Town03", "Town04"],
        seq_len=16,
        stride=4,
    )

    print(f"Dataset length: {len(ds)}")

    sample = ds[0]
    print(f"\nSample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")

    r = sample["rewards"]
    print(f"\nReward stats: min={r.min():.3f}, max={r.max():.3f}, "
          f"mean={r.mean():.3f}, std={r.std():.3f}")
    print(f"Speed range:  [{sample['speeds'].min():.2f}, {sample['speeds'].max():.2f}] m/s")

    # Check reward has actual variance
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    all_rewards = batch["rewards"].flatten()
    print(f"\nBatch reward stats: mean={all_rewards.mean():.3f}, std={all_rewards.std():.3f}")
    assert all_rewards.std() > 0.05, f"Reward variance too low: {all_rewards.std():.4f}"

    print("\nSequenceDataset v2 test PASSED")
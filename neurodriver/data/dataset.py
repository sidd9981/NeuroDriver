"""
CARLA Driving Dataset — supports TransFuser and TCP data formats.

TransFuser format (folder structure):
    data_root/
    ├── Town01_Rep0_route_002516_route0_01_08_10_14_49/
    │   ├── rgb/
    │   │   ├── 0000.png
    │   │   ├── 0001.png
    │   │   └── ...
    │   └── measurements/
    │       ├── 0000.json
    │       ├── 0001.json
    │       └── ...
    ├── Town01_Rep0_route_002518_route0_01_10_03_16_03/
    │   └── ...
    └── ...

Each measurements/*.json file contains:
    {
        "x": float,              # ego position x
        "y": float,              # ego position y
        "theta": float,          # ego heading (radians)
        "speed": float,          # ego speed (m/s)
        "steer": float,          # steering angle [-1, 1]
        "throttle": float,       # throttle [0, 1]
        "brake": float,          # brake [0, 1]
        "x_command": float,      # next waypoint x
        "y_command": float,      # next waypoint y
        "command": int,           # high-level: 1=left, 2=right, 3=straight, 4=follow
        ...
    }

TCP format:
    Each route folder has a packed_data.npy containing all measurements,
    plus rgb/ folder with images referenced by path in the npy.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from neurodriver.data.transforms import get_train_transforms, get_val_transforms


class TransFuserDataset(Dataset):
    """
    Loads driving data from the TransFuser dataset format.
    
    Args:
        data_root: Path to dataset root directory.
        towns: List of town names to include (e.g., ["Town01", "Town02"]).
        image_size: Tuple (H, W) for resized images.
        augment: Whether to apply data augmentation.
        seq_len: Number of consecutive frames to return (1 = single frame).
    """
    
    def __init__(
        self,
        data_root: str,
        towns: list[str] | None = None,
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
        seq_len: int = 1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.seq_len = seq_len
        
        # Set up transforms
        if augment:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)
        
        # Discover all valid (image, measurement) pairs
        self.samples = []
        self._discover_samples(towns)
        
        print(f"TransFuserDataset: Found {len(self.samples)} samples "
              f"from {len(towns) if towns else 'all'} towns")
    
    def _discover_samples(self, towns: list[str] | None):
        """
        Walk through dataset folders and find all valid frame pairs.
        
        A frame is valid if both rgb/{frame_id}.png and 
        measurements/{frame_id}.json exist.
        """
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.data_root}\n"
                f"Please download the dataset first. See scripts/download_data.sh"
            )
        
        # Find all route folders
        route_dirs = sorted(self.data_root.iterdir())
        
        for route_dir in route_dirs:
            if not route_dir.is_dir():
                continue
            
            # Filter by town if specified
            if towns is not None:
                # Route folder names typically contain the town name
                route_name_lower = route_dir.name.lower()
                if not any(t.lower() in route_name_lower for t in towns):
                    continue
            
            rgb_dir = route_dir / "rgb"
            meas_dir = route_dir / "measurements"
            
            if not rgb_dir.exists() or not meas_dir.exists():
                # Try alternate naming: some datasets use "rgb_front" 
                rgb_dir = route_dir / "rgb_front"
                if not rgb_dir.exists():
                    continue
            
            # Get sorted frame IDs that have BOTH image and measurement
            # Support both .png (TransFuser) and .jpg (LEAD) images
            rgb_files = {}
            for ext in ("*.png", "*.jpg"):
                for f in rgb_dir.glob(ext):
                    rgb_files[f.stem] = f.name  # stem -> full filename with ext
            
            meas_files = {f.stem for f in meas_dir.glob("*.json")}
            valid_frames = sorted(rgb_files.keys() & meas_files)
            
            for frame_id in valid_frames:
                self.samples.append({
                    "rgb_path": str(rgb_dir / rgb_files[frame_id]),
                    "meas_path": str(meas_dir / f"{frame_id}.json"),
                    "route": route_dir.name,
                    "frame_id": frame_id,
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single training sample.
        
        Returns dict with keys:
            image:     (3, H, W) float tensor, normalized to [0, 1]
            speed:     (1,) float tensor, ego speed in m/s
            command:   (1,) long tensor, high-level command {1,2,3,4}
            target_point: (2,) float tensor, next waypoint [x, y] in ego frame
            steer:     (1,) float tensor, [-1, 1]
            throttle:  (1,) float tensor, [0, 1]
            brake:     (1,) float tensor, [0, 1]
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample["rgb_path"]).convert("RGB")
        img = self.transform(img)  # Returns (3, H, W) tensor
        
        # Load measurements
        with open(sample["meas_path"], "r") as f:
            meas = json.load(f)
        
        # Extract controls (what the expert did)
        steer = float(meas.get("steer", 0.0))
        throttle = float(meas.get("throttle", 0.0))
        brake = float(meas.get("brake", 0.0))
        
        # Extract state info
        speed = float(meas.get("speed", 0.0))
        
        # High-level command: 1=left, 2=right, 3=straight, 4=follow lane
        # Some datasets use "command", others use "target_command"
        command = int(meas.get("command", meas.get("target_command", 4)))
        
        # Target waypoint in ego vehicle frame
        # This is the next GPS waypoint the car should drive toward
        x_cmd = float(meas.get("x_command", meas.get("x_target", 0.0)))
        y_cmd = float(meas.get("y_command", meas.get("y_target", 0.0)))
        
        # Convert to ego-relative coordinates if absolute coords given
        # (Some datasets provide absolute world coords, others ego-relative)
        ego_x = float(meas.get("x", 0.0))
        ego_y = float(meas.get("y", 0.0))
        theta = float(meas.get("theta", 0.0))
        
        # Transform target point to ego frame
        dx = x_cmd - ego_x
        dy = y_cmd - ego_y
        target_x = dx * np.cos(theta) + dy * np.sin(theta)
        target_y = -dx * np.sin(theta) + dy * np.cos(theta)
        
        return {
            "image": img,
            "speed": torch.tensor([speed], dtype=torch.float32),
            "command": torch.tensor([command], dtype=torch.long),
            "target_point": torch.tensor([target_x, target_y], dtype=torch.float32),
            "steer": torch.tensor([steer], dtype=torch.float32),
            "throttle": torch.tensor([throttle], dtype=torch.float32),
            "brake": torch.tensor([brake], dtype=torch.float32),
        }


class TCPDataset(Dataset):
    """
    Loads driving data from the TCP packed_data.npy format.
    
    TCP stores all measurements in a single .npy file per route,
    with image paths referenced inside the npy dict.
    
    This is a lighter format (~50GB vs 210GB for TransFuser).
    
    Args:
        data_root: Path to TCP dataset root.
        towns: List of town names to filter.
        image_size: Tuple (H, W) for resized images.
        augment: Whether to apply data augmentation.
    """
    
    def __init__(
        self,
        data_root: str,
        towns: list[str] | None = None,
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        
        if augment:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)
        
        # Load all packed_data.npy files
        self.front_imgs = []
        self.speeds = []
        self.steers = []
        self.throttles = []
        self.brakes = []
        self.commands = []
        self.target_xs = []
        self.target_ys = []
        
        self._load_data(towns)
        
        print(f"TCPDataset: Loaded {len(self.front_imgs)} samples")
    
    def _load_data(self, towns: list[str] | None):
        """Load packed_data.npy files from each route folder."""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")
        
        for route_dir in sorted(self.data_root.iterdir()):
            if not route_dir.is_dir():
                continue
            
            if towns is not None:
                route_lower = route_dir.name.lower()
                if not any(t.lower() in route_lower for t in towns):
                    continue
            
            npy_path = route_dir / "packed_data.npy"
            if not npy_path.exists():
                continue
            
            data = np.load(str(npy_path), allow_pickle=True).item()
            
            n_frames = len(data.get("front_img", []))
            for i in range(n_frames):
                self.front_imgs.append(data["front_img"][i])
                self.speeds.append(data["speed"][i])
                self.steers.append(data["action"][i][0] if "action" in data else data.get("steer", [0.0])[i])
                self.throttles.append(data["action"][i][1] if "action" in data else data.get("throttle", [0.0])[i])
                self.brakes.append(data.get("only_ap_brake", [0.0])[i])
                self.commands.append(data.get("command", [4])[i] if "command" in data else 4)
                self.target_xs.append(data.get("x_command", [0.0])[i] if "x_command" in data else 0.0)
                self.target_ys.append(data.get("y_command", [0.0])[i] if "y_command" in data else 0.0)
    
    def __len__(self) -> int:
        return len(self.front_imgs)
    
    def __getitem__(self, idx: int) -> dict:
        # Load image — TCP stores paths relative to data_root
        img_path = self.front_imgs[idx]
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        
        full_path = self.data_root / img_path if not os.path.isabs(img_path) else img_path
        img = Image.open(str(full_path)).convert("RGB")
        img = self.transform(img)
        
        return {
            "image": img,
            "speed": torch.tensor([self.speeds[idx]], dtype=torch.float32),
            "command": torch.tensor([self.commands[idx]], dtype=torch.long),
            "target_point": torch.tensor(
                [self.target_xs[idx], self.target_ys[idx]], dtype=torch.float32
            ),
            "steer": torch.tensor([self.steers[idx]], dtype=torch.float32),
            "throttle": torch.tensor([self.throttles[idx]], dtype=torch.float32),
            "brake": torch.tensor([self.brakes[idx]], dtype=torch.float32),
        }


# Factory function

def build_dataset(cfg: dict, split: str = "train") -> Dataset:
    """
    Build the appropriate dataset from config.
    
    Args:
        cfg: The full config dict (or OmegaConf).
        split: "train" or "val".
    
    Returns:
        A PyTorch Dataset instance.
    """
    data_cfg = cfg["data"]
    dataset_type = data_cfg["dataset_type"]
    
    towns = data_cfg["train_towns"] if split == "train" else data_cfg["val_towns"]
    augment = data_cfg["augment"] and split == "train"  # Only augment training data
    image_size = tuple(data_cfg["image_size"])
    
    if dataset_type == "transfuser":
        return TransFuserDataset(
            data_root=data_cfg["data_root"],
            towns=towns,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_type == "tcp":
        return TCPDataset(
            data_root=data_cfg["data_root"],
            towns=towns,
            image_size=image_size,
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'transfuser' or 'tcp'.")


# Quick test

if __name__ == "__main__":
    """
    Quick test: Create a small dummy dataset and verify everything works.
    Run: python -m neurodriver.data.dataset
    """
    import tempfile
    
    print("Creating dummy TransFuser dataset for testing...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy route folder
        route_dir = Path(tmpdir) / "routes_town01_test_00"
        (route_dir / "rgb").mkdir(parents=True)
        (route_dir / "measurements").mkdir(parents=True)
        
        # Create 10 dummy frames
        for i in range(10):
            # Dummy image (random colored rectangle)
            img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
            img.save(route_dir / "rgb" / f"{i:04d}.png")
            
            # Dummy measurements
            meas = {
                "x": float(i * 2.0),
                "y": 0.0,
                "theta": 0.0,
                "speed": 5.0 + float(i) * 0.5,
                "steer": np.random.uniform(-0.3, 0.3),
                "throttle": np.random.uniform(0.3, 0.7),
                "brake": 0.0,
                "x_command": float(i * 2.0 + 10.0),
                "y_command": 0.0,
                "command": np.random.choice([1, 2, 3, 4]),
            }
            with open(route_dir / "measurements" / f"{i:04d}.json", "w") as f:
                json.dump(meas, f)
        
        # Test the dataset
        ds = TransFuserDataset(
            data_root=tmpdir,
            towns=["Town01"],
            image_size=(256, 256),
            augment=False,
        )
        
        print(f"Dataset length: {len(ds)}")
        
        sample = ds[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"  image:        shape={sample['image'].shape}, dtype={sample['image'].dtype}")
        print(f"  speed:        {sample['speed'].item():.2f} m/s")
        print(f"  command:      {sample['command'].item()}")
        print(f"  target_point: {sample['target_point'].tolist()}")
        print(f"  steer:        {sample['steer'].item():.3f}")
        print(f"  throttle:     {sample['throttle'].item():.3f}")
        print(f"  brake:        {sample['brake'].item():.3f}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
        
        print("\n Dataset test PASSED!")
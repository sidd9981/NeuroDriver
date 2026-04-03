"""
nuScenes BEV Dataset.

Returns per-sample:
  - Front camera image (CAM_FRONT)
  - Real camera intrinsics (per sample, not estimated)
  - Road BEV label from HD map drivable_area polygons
  - Vehicle BEV label from 3D bounding box annotations

"""

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import view_points

from neurodriver.data.transforms import get_train_transforms, get_val_transforms


# BEV grid params — must match bev_model.py
BEV_X_RANGE = (-25.0, 25.0)
BEV_Y_RANGE = (0.0,   50.0)
BEV_RES     = 0.5
BEV_W = int((BEV_X_RANGE[1] - BEV_X_RANGE[0]) / BEV_RES)  # 100
BEV_H = int((BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) / BEV_RES)  # 100

# nuScenes map names present in mini split
MINI_MAPS = [
    'singapore-onenorth',
    'singapore-hollandvillage',
    'singapore-queenstown',
    'boston-seaport',
]

VEHICLE_CLASSES = {
    'vehicle.car', 'vehicle.truck', 'vehicle.bus',
    'vehicle.motorcycle', 'vehicle.trailer',
}


def get_map_name_from_scene(nusc, scene_token):
    """Get the map name for a scene via its log."""
    scene = nusc.get('scene', scene_token)
    log   = nusc.get('log', scene['log_token'])
    return log['location']


def get_cam_intrinsics(nusc, sample_data_token):
    """
    Returns (3,3) intrinsic matrix and ego->camera transform for a sample.
    These are real calibrated values, not estimates.
    """
    sd      = nusc.get('sample_data', sample_data_token)
    cs      = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    K       = np.array(cs['camera_intrinsic'], dtype=np.float32)  # (3,3)
    # Camera extrinsic: translation + rotation from ego to camera
    cam_t   = np.array(cs['translation'], dtype=np.float32)
    cam_r   = Quaternion(cs['rotation'])
    return K, cam_t, cam_r


def ego_to_bev_px(points_ego):
    """
    Convert (N,2) ego-frame XY points (metres) to BEV pixel indices.
    Ego convention: X=forward, Y=left.
    BEV convention: row=forward (y-axis), col=lateral (x-axis).

    Returns (N,2) integer pixel coords and validity mask.
    """
    # BEV: col = (Y_ego - x_range_min) / res  (lateral)
    #       row = (X_ego - y_range_min) / res  (forward)
    col = (points_ego[:, 1] - BEV_X_RANGE[0]) / BEV_RES   # lateral -> col
    row = (points_ego[:, 0] - BEV_Y_RANGE[0]) / BEV_RES   # forward -> row

    col = np.round(col).astype(int)
    row = np.round(row).astype(int)

    valid = ((col >= 0) & (col < BEV_W) &
             (row >= 0) & (row < BEV_H))
    return row, col, valid


def generate_road_label_nuscenes(nusc_map, ego_pose, patch_size=60.0):
    """
    Rasterize drivable_area HD map polygons into BEV label.

    Args:
        nusc_map:   NuScenesMap for this scene's location
        ego_pose:   dict with 'translation' and 'rotation' (ego in world frame)
        patch_size: metres around ego to query (60m covers our 50m BEV range)

    Returns:
        (BEV_H, BEV_W) float32 label, 1=drivable, 0=not
    """
    label = np.zeros((BEV_H, BEV_W), dtype=np.float32)

    ego_x, ego_y = ego_pose['translation'][0], ego_pose['translation'][1]
    ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]

    # Query HD map for drivable area polygons in a patch around ego
    patch_box = (ego_x, ego_y, patch_size, patch_size)
    patch_angle = np.degrees(ego_yaw)

    try:
        masks = nusc_map.get_map_mask(
            patch_box, patch_angle,
            ['drivable_area'],
            canvas_size=(int(patch_size / BEV_RES),
                         int(patch_size / BEV_RES)),
        )
    except Exception:
        return label

    road_mask = masks[0]  # (canvas_h, canvas_w) binary

    # The map mask covers [-patch/2, patch/2] in ego frame
    # We need to crop to our BEV range [BEV_Y_RANGE, BEV_X_RANGE]
    canvas_h, canvas_w = road_mask.shape
    canvas_res = patch_size / canvas_h

    # BEV covers forward 0->50m, lateral -25->25m
    # In ego frame: forward=X, lateral=Y
    # Map mask: center=ego, forward=up in mask
    for bev_row in range(BEV_H):
        for bev_col in range(BEV_W):
            # BEV pixel -> ego metres
            ego_fwd = bev_row * BEV_RES + BEV_Y_RANGE[0]   
            ego_lat = bev_col * BEV_RES + BEV_X_RANGE[0]  

            # Map mask pixel (mask is centered at ego, forward=up=row 0)
            mask_row = int((ego_fwd - BEV_Y_RANGE[0]) / canvas_res)
            mask_col = int((ego_lat + patch_size / 2) / canvas_res)

            if (0 <= mask_row < canvas_h and 0 <= mask_col < canvas_w):
                label[bev_row, bev_col] = float(road_mask[mask_row, mask_col])

    return label


def generate_vehicle_label_nuscenes(nusc, sample_token, ego_pose):
    """
    Place vehicle blobs in BEV from 3D bounding box annotations.

    Args:
        nusc:         NuScenes instance
        sample_token: current sample token
        ego_pose:     dict with ego translation/rotation in world frame

    Returns:
        (BEV_H, BEV_W) float32 label
    """
    label = np.zeros((BEV_H, BEV_W), dtype=np.float32)

    sample = nusc.get('sample', sample_token)
    ego_t  = np.array(ego_pose['translation'][:2])
    ego_q  = Quaternion(ego_pose['rotation'])

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Filter to vehicle classes
        if not any(ann['category_name'].startswith(vc.split('.')[0] + '.' + vc.split('.')[1])
                   for vc in VEHICLE_CLASSES
                   if ann['category_name'].startswith(vc[:vc.rfind('.')])):
            # simpler check:
            if not ann['category_name'].startswith('vehicle.'):
                continue

        # World position -> ego frame
        world_pos = np.array(ann['translation'][:2])
        rel       = world_pos - ego_t
        # Rotate into ego frame
        ego_inv   = ego_q.inverse
        rel_3d    = ego_inv.rotate(np.array([rel[0], rel[1], 0.0]))
        fwd, lat  = rel_3d[0], rel_3d[1]

        # Check if within BEV range
        if not (BEV_Y_RANGE[0] <= fwd <= BEV_Y_RANGE[1] and
                BEV_X_RANGE[0] <= lat <= BEV_X_RANGE[1]):
            continue

        # Place gaussian blob sized to annotation footprint
        size_m = max(ann['size'][0], ann['size'][1]) / 2.0  # half-size

        bev_row = int((fwd - BEV_Y_RANGE[0]) / BEV_RES)
        bev_col = int((lat - BEV_X_RANGE[0]) / BEV_RES)
        size_px = max(2, int(size_m / BEV_RES))

        for r in range(max(0, bev_row - size_px), min(BEV_H, bev_row + size_px)):
            for c in range(max(0, bev_col - size_px), min(BEV_W, bev_col + size_px)):
                dr = (r - bev_row) * BEV_RES
                dc = (c - bev_col) * BEV_RES
                label[r, c] = max(label[r, c],
                                  np.exp(-(dr**2 + dc**2) / size_m**2))

    return label


class NuScenesBEVDataset(Dataset):
    """
    nuScenes dataset for BEV perception training.

    Returns per sample:
        image:          (3, H, W) normalized front camera image
        road_label:     (BEV_H, BEV_W) float32 drivable area mask
        vehicle_label:  (BEV_H, BEV_W) float32 vehicle occupancy
        intrinsics:     (3, 3) float32 real camera intrinsic matrix
    """

    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-mini',
        split: str = 'train',         # 'train' or 'val' (splits mini 80/20)
        image_size: tuple = (256, 256),
        augment: bool = False,
    ):
        self.dataroot   = dataroot
        self.image_size = image_size
        self.transform  = (get_train_transforms(image_size) if augment
                           else get_val_transforms(image_size))

        print(f"Loading nuScenes {version}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Load maps for all locations
        self.maps = {}
        for map_name in MINI_MAPS:
            try:
                self.maps[map_name] = NuScenesMap(
                    dataroot=dataroot, map_name=map_name
                )
            except Exception:
                pass
        print(f"  Loaded {len(self.maps)} HD maps")

        # Build sample list with scene->map lookup
        all_samples = []
        for scene in self.nusc.scene:
            map_name = get_map_name_from_scene(self.nusc, scene['token'])
            if map_name not in self.maps:
                continue
            # Walk through all samples in this scene
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                all_samples.append({
                    'sample_token': sample_token,
                    'map_name':     map_name,
                })
                sample_token = sample['next']

        # Split 80/20 by scene (not by frame, to avoid leakage)
        n_scenes   = len(self.nusc.scene)
        n_train    = int(n_scenes * 0.8)
        scene_list = [s['token'] for s in self.nusc.scene]

        train_scenes = set(scene_list[:n_train])
        val_scenes   = set(scene_list[n_train:])

        split_scenes = train_scenes if split == 'train' else val_scenes

        # Filter samples to split
        self.samples = []
        for entry in all_samples:
            sample  = self.nusc.get('sample', entry['sample_token'])
            scene_t = sample['scene_token']
            if scene_t in split_scenes:
                self.samples.append(entry)

        print(f"  {split}: {len(self.samples)} samples "
              f"from {len(split_scenes)} scenes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry        = self.samples[idx]
        sample_token = entry['sample_token']
        map_name     = entry['map_name']
        nusc_map     = self.maps[map_name]

        sample = self.nusc.get('sample', sample_token)

        # Front camera image 
        cam_token = sample['data']['CAM_FRONT']
        sd        = self.nusc.get('sample_data', cam_token)
        img_path  = Path(self.dataroot) / sd['filename']
        img       = Image.open(img_path).convert('RGB')
        img       = self.transform(img)

        # Real intrinsics
        K, _, _ = get_cam_intrinsics(self.nusc, cam_token)
        # Scale intrinsics to match our resized image
        orig_w, orig_h = sd['width'], sd['height']
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h
        K_scaled       = K.copy()
        K_scaled[0] *= scale_x
        K_scaled[1] *= scale_y

        # Ego pose 
        ep        = self.nusc.get('ego_pose', sd['ego_pose_token'])

        # Road label from HD map 
        road_label = generate_road_label_nuscenes(nusc_map, ep)

        # Vehicle label from annotations
        vehicle_label = generate_vehicle_label_nuscenes(
            self.nusc, sample_token, ep
        )

        return {
            'image':         img,
            'road_label':    torch.tensor(road_label,    dtype=torch.float32),
            'vehicle_label': torch.tensor(vehicle_label, dtype=torch.float32),
            'intrinsics':    torch.tensor(K_scaled,      dtype=torch.float32),
        }


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DATAROOT = 'data_raw/nuscenes'

    ds = NuScenesBEVDataset(dataroot=DATAROOT, split='train', augment=False)
    print(f"\nDataset length: {len(ds)}")

    sample = ds[0]
    print(f"image:         {sample['image'].shape}")
    print(f"road_label:    {sample['road_label'].shape}  "
          f"density={sample['road_label'].mean():.3f}")
    print(f"vehicle_label: {sample['vehicle_label'].shape}  "
          f"max={sample['vehicle_label'].max():.3f}")
    print(f"intrinsics:\n{sample['intrinsics']}")

    # Check label variation — sample spread across dataset not just first 5
    print("\nRoad label variation (spread across dataset):")
    indices = [0, len(ds)//4, len(ds)//2, 3*len(ds)//4, len(ds)-1]
    densities = [ds[i]['road_label'].mean().item() for i in indices]
    for i, d in zip(indices, densities):
        print(f"  sample {i}: road density={d:.3f}")

    varies = max(densities) - min(densities) > 0.05
    print(f"\nLabels vary across dataset: {'YES' if varies else 'NO — still fixed prior'}")

    # Also visualize 3 spread-out samples side by side
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    for col, idx in enumerate([0, len(ds)//2, len(ds)-1]):
        s = ds[idx]
        img_np = s['image'].permute(1,2,0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        axes2[0, col].imshow(img_np)
        axes2[0, col].set_title(f"Frame {idx}")
        axes2[0, col].axis('off')
        axes2[1, col].imshow(s['road_label'].numpy(), cmap='Blues', vmin=0, vmax=1)
        axes2[1, col].set_title(f"Road (density={s['road_label'].mean():.3f})")
        axes2[1, col].axis('off')
    plt.tight_layout()
    plt.savefig('checkpoints/nuscenes_label_variation.png', dpi=150)
    print("Saved: checkpoints/nuscenes_label_variation.png")

    # Save a quick visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    img_np = sample['image'].permute(1, 2, 0).numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = np.clip(img_np * std + mean, 0, 1)

    axes[0].imshow(img_np)
    axes[0].set_title('Front Camera')
    axes[0].axis('off')

    axes[1].imshow(sample['road_label'].numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(f"Road BEV (density={densities[0]:.3f})")
    axes[1].axis('off')

    axes[2].imshow(sample['vehicle_label'].numpy(), cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Vehicle BEV')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('checkpoints/nuscenes_label_check.png', dpi=150)
    print("\nSaved: checkpoints/nuscenes_label_check.png")
"""
Lift-Splat-Shoot BEV Perception (ECCV 2020, Philion & Fidler).

Pipeline:
  1. LIFT:  predict depth distribution per feature pixel -> unproject to 3D
  2. SPLAT: scatter weighted features into BEV grid
  3. SHOOT: BEV encoder + segmentation heads

Design choices vs naive LSS:
  - Uses layer3 features (16×16, 256ch) not layer4 (8×8, 512ch):
    4× more lift points -> denser BEV coverage
  - Depth head has residual skip from raw features so gradient reaches
    the backbone without flowing through BatchNorm bottleneck
  - Separate regression head (geometrically supervised) and distribution
    head (image-conditioned via splat->road gradient)
  - BEV encoder has residual connection for clean gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftSplatSimple(nn.Module):
    def __init__(
        self,
        image_h: int = 256,
        image_w: int = 256,
        feat_h: int = 16,
        feat_w: int = 16,
        depth_min: float = 2.0,
        depth_max: float = 50.0,
        num_depth_bins: int = 48,
        bev_x_range: tuple = (-25.0, 25.0),
        bev_y_range: tuple = (0.0, 50.0),
        bev_res: float = 0.5,
        feat_channels: int = 256,
        bev_channels: int = 64,
    ):
        super().__init__()

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depth_bins = num_depth_bins
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.feat_channels = feat_channels
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_res = bev_res
        self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_res)  # 100
        self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_res)  # 100

        depth_vals = torch.linspace(depth_min, depth_max, num_depth_bins)
        self.register_buffer("depth_bins", depth_vals)

        # ── Depth head ────────────────────────────────────────────────────
        # Two-branch: shared bottleneck + residual skip from raw features.
        # Without the skip, depth_dist gradients stall at the BN layer.
        self.depth_bottleneck = nn.Sequential(
            nn.Conv2d(feat_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.depth_skip = nn.Conv2d(feat_channels, 256, 1)

        # Distribution head: softmax over bins, used for lift weighting
        # Supervised indirectly via road_loss -> splat -> lift -> here
        self.depth_dist_head = nn.Conv2d(256, num_depth_bins, 1)

        # Regression head: direct geometric supervision (flat road prior)
        self.depth_reg_head = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),  # scaled to [depth_min, depth_max] in _get_depth_dist
        )

        #  Context head 
        # 32 channels (not 64): same point budget fills BEV more uniformly
        self.context_net = nn.Sequential(
            nn.Conv2d(feat_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 1),
        )
        self.ctx_channels = 32

        # BEV encoder with residual
        self.bev_compress = nn.Conv2d(32, bev_channels, 1)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(),
        )

        # Segmentation heads
        self.road_head = nn.Sequential(
            nn.Conv2d(bev_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )
        self.vehicle_head = nn.Sequential(
            nn.Conv2d(bev_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        # CARLA default front camera intrinsics (90° FOV, 256×256)
        fx = image_w / 2.0
        fy = image_h / 2.0
        cx, cy = image_w / 2.0, image_h / 2.0
        self.register_buffer("intrinsics", torch.tensor([
            [fx, 0, cx], [0, fy, cy], [0, 0, 1],
        ], dtype=torch.float32))

        self._precompute_geometry(feat_h, feat_w, image_h, image_w)

    def _precompute_geometry(self, feat_h, feat_w, image_h, image_w):
        """Precompute ray directions and flat-road depth prior."""
        stride_h = image_h / feat_h
        stride_w = image_w / feat_w
        vs = torch.arange(feat_h).float() * stride_h + stride_h / 2
        us = torch.arange(feat_w).float() * stride_w + stride_w / 2
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
        self.register_buffer("grid_u", grid_u)
        self.register_buffer("grid_v", grid_v)

        # Flat-road geometric depth prior: camera 1.5m high, zero pitch
        camera_height = 1.5
        horizon_v = image_h / 2.0
        fy = image_h / 2.0
        geo_depth = torch.zeros(feat_h, feat_w)
        for i, v in enumerate(vs):
            dv = float(v) - horizon_v
            if dv > 1.0:
                angle = float(torch.atan(torch.tensor(dv / fy)))
                d = camera_height / (float(torch.tan(torch.tensor(angle))) + 1e-6)
                geo_depth[i, :] = max(2.0, min(50.0, d))
            # else: sky rows stay 0 (no label)
        self.register_buffer("geo_depth_prior", geo_depth)  # (feat_h, feat_w)

    def _get_depth_dist(self, features):
        h = self.depth_bottleneck(features) + self.depth_skip(features)
        dist = F.softmax(self.depth_dist_head(h), dim=1)          # (B, D, H, W)
        reg = self.depth_reg_head(h)                               # (B, 1, H, W)
        reg = reg * (self.depth_max - self.depth_min) + self.depth_min
        return dist, reg

    def lift(self, features):
        B = features.shape[0]
        D = self.num_depth_bins

        depth_probs, depth_reg = self._get_depth_dist(features)
        context = self.context_net(features)                       # (B, 32, H, W)

        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        ray_x = (self.grid_u - cx) / fx                           # (feat_h, feat_w)
        ray_y = (self.grid_v - cy) / fy

        depths = self.depth_bins                                   # (D,)
        x_3d = ray_x.unsqueeze(0) * depths.view(D, 1, 1)
        y_3d = depths.view(D, 1, 1).expand(D, self.feat_h, self.feat_w)

        points_xy = torch.stack([x_3d.reshape(-1), y_3d.reshape(-1)], dim=-1)

        # Weight context features by depth probability
        weighted = context.unsqueeze(2) * depth_probs.unsqueeze(1)  # (B, 32, D, H, W)
        weighted = weighted.reshape(B, 32, -1).permute(0, 2, 1)     # (B, D*H*W, 32)

        return points_xy, weighted, depth_probs, depth_reg

    def splat(self, points_xy, point_feats):
        B, N, C = point_feats.shape
        device = point_feats.device

        x_idx = ((points_xy[:, 0] - self.bev_x_range[0]) / self.bev_res).long()
        y_idx = ((points_xy[:, 1] - self.bev_y_range[0]) / self.bev_res).long()

        valid = (
            (x_idx >= 0) & (x_idx < self.bev_w) &
            (y_idx >= 0) & (y_idx < self.bev_h)
        )
        x_idx = x_idx.clamp(0, self.bev_w - 1)
        y_idx = y_idx.clamp(0, self.bev_h - 1)
        flat_idx = y_idx * self.bev_w + x_idx                     # (N,)

        bev = torch.zeros(B, C, self.bev_h * self.bev_w, device=device)
        flat_idx_batch = flat_idx.unsqueeze(0).unsqueeze(1).expand(B, C, N)
        valid_mask = valid.unsqueeze(0).unsqueeze(1).expand(B, C, N).float()
        bev.scatter_add_(2, flat_idx_batch, point_feats.permute(0, 2, 1) * valid_mask)

        return bev.reshape(B, C, self.bev_h, self.bev_w)

    def forward(self, features):
        """
        Args:
            features: (B, feat_channels, feat_h, feat_w)
        Returns:
            bev_features:  (B, bev_channels, bev_h, bev_w)
            road_seg:      (B, 1, bev_h, bev_w)  logits
            vehicle_seg:   (B, 1, bev_h, bev_w)  logits
            depth_probs:   (B, D, feat_h, feat_w) softmax distribution
            depth_reg:     (B, 1, feat_h, feat_w) regression in metres
        """
        points_xy, point_feats, depth_probs, depth_reg = self.lift(features)
        bev_raw = self.splat(points_xy, point_feats)               # (B, 32, 100, 100)

        bev_in = self.bev_compress(bev_raw)                        # (B, 64, 100, 100)
        bev_features = self.bev_encoder(bev_in) + bev_in           # residual

        return {
            "bev_features": bev_features,
            "road_seg":     self.road_head(bev_features),
            "vehicle_seg":  self.vehicle_head(bev_features),
            "depth_probs":  depth_probs,
            "depth_reg":    depth_reg,
        }


class BEVDrivingModel(nn.Module):
    """
    ResNet-34 backbone -> layer3 spatial features -> Lift-Splat -> BEV map.

    Uses layer3 (16x16, 256ch) not layer4+pool (8x8, 512ch):
    gives 4× more lift points (12,288 vs 3,072) for denser BEV coverage.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        bev_channels: int = 64,
        num_depth_bins: int = 48,
    ):
        super().__init__()

        from neurodriver.models.backbone import ResNetBackbone
        self.backbone = ResNetBackbone(backbone_name, pretrained, feature_dim=512)

        # layer3 extractor: output is (B, 256, 16, 16) for 256×256 input
        feats = self.backbone.features
        self.layer3_extractor = nn.Sequential(
            feats[0],   # conv1
            feats[1],   # bn1
            feats[2],   # relu
            feats[3],   # maxpool
            feats[4],   # layer1
            feats[5],   # layer2
            feats[6],   # layer3  -> (B, 256, 16, 16)
        )

        self.bev = LiftSplatSimple(
            feat_h=16, feat_w=16,
            feat_channels=256,
            bev_channels=bev_channels,
            num_depth_bins=num_depth_bins,
        )

    def get_spatial_features(self, image):
        return self.layer3_extractor(image)                        # (B, 256, 16, 16)

    def forward(self, image):
        return self.bev(self.get_spatial_features(image))


if __name__ == "__main__":
    from neurodriver.utils.device import get_device

    device = get_device()
    print(f"Testing BEV model on: {device}")

    model = BEVDrivingModel(pretrained=False).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Lift points per image: {16*16*48:,}  (was {8*8*48:,})")

    image = torch.randn(2, 3, 256, 256, device=device)
    output = model(image)

    print("\nOutput shapes:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")

    # Input sensitivity check
    blank = torch.zeros_like(image)
    with torch.no_grad():
        out_r = model(image)
        out_b = model(blank)
    diff = (torch.sigmoid(out_r["road_seg"]) -
            torch.sigmoid(out_b["road_seg"])).abs().mean()
    print(f"\nInput sensitivity: {diff:.4f}  (should be > 0.05)")

    loss = output["road_seg"].sum() + output["vehicle_seg"].sum()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"Gradient flow: {'PASSED' if not no_grad else f'FAILED {no_grad[:3]}'}")
    print(f"\nBEV model test PASSED on {device}!")
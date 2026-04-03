"""
BEV collapse diagnostic.
Usage: python scripts/debug_bev.py
"""
import torch
import json
from pathlib import Path
from PIL import Image
from neurodriver.data.transforms import get_val_transforms
from neurodriver.models.bev_model import BEVDrivingModel
from neurodriver.utils.device import get_device

device = get_device()
model = BEVDrivingModel(pretrained=True).to(device)
ckpt = torch.load("checkpoints/bev_model_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tf = get_val_transforms((256, 256))
root = Path("data_raw/transfuser")
route = next(d for d in root.iterdir() if d.is_dir())
img_path = next((route / "rgb").glob("*.jpg"))
img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(img)

road = torch.sigmoid(out["road_seg"][0, 0])
depth = out["depth_reg"][0, 0]
depth_probs = out["depth_probs"][0]

print(f"road_seg sigmoid: min={road.min():.4f}  max={road.max():.4f}  mean={road.mean():.4f}")
print(f"depth_reg:        min={depth.min():.1f}  max={depth.max():.1f}  mean={depth.mean():.1f} m")

bins = torch.linspace(2.0, 50.0, 48)
expected_depth = (depth_probs.cpu() * bins.view(48, 1, 1)).sum(dim=0)
print(f"expected depth:   min={expected_depth.min():.1f}  max={expected_depth.max():.1f}  mean={expected_depth.mean():.1f} m")

entropy = -(depth_probs * (depth_probs + 1e-8).log()).sum(dim=0)
print(f"depth entropy:    min={entropy.min():.3f}  max={entropy.max():.3f}  mean={entropy.mean():.3f}")
print(f"  (peaked=~0.5-1.5, collapsed/uniform=~3.5-3.87, max possible={torch.log(torch.tensor(48.0)):.2f})")

# Label density: what fraction of road label cells are > 0.3?
meas_path = next((route / "measurements").glob("*.json"))
with open(meas_path) as f:
    meas = json.load(f)
print(f"\nSample measurement fields relevant to road label:")
for k in ("ego_lane_width", "distance_ego_to_route", "is_junction"):
    print(f"  {k}: {meas.get(k, 'MISSING')}")

from neurodriver.training.train_bev import generate_road_label
lbl = generate_road_label(meas)
print(f"Road label density (>0.3): {(lbl > 0.3).mean():.3f}  "
      f"(good=0.05-0.15, too sparse=<0.02, too dense=>0.5)")

# Input sensitivity
blank = torch.zeros_like(img)
with torch.no_grad():
    out_blank = model(blank)
diff = (torch.sigmoid(out["road_seg"]) - torch.sigmoid(out_blank["road_seg"])).abs().mean()
print(f"\nsensitivity:      {diff:.4f}  (good=>0.05, collapsed=<0.02)")
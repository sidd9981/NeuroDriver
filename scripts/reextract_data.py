"""
Re-extract LEAD dataset pickles to include all reward-relevant fields.


Usage:
    python scripts/reextract_rich_data.py
"""

import json
import lzma
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np


# HuggingFace cache location for the LEAD dataset
HF_CACHE = Path.home() / ".cache/huggingface/hub/datasets--ln2697--lead_carla"
OUTPUT_DIR = Path("data_raw/transfuser")


def find_cached_zips():
    """Find all cached zip files from the LEAD dataset."""
    if not HF_CACHE.exists():
        print(f"HuggingFace cache not found at {HF_CACHE}")
        return []

    zips = list(HF_CACHE.rglob("*.zip"))
    print(f"Found {len(zips)} cached zip files")
    return sorted(zips)


def extract_rich_measurements(meta):
    """
    Extract a rich measurement dict from a LEAD pickle metadata dict.

    Based on fields used by:
      - Roach (ICCV 2021): speed vs target_speed, lateral deviation, hazards
      - CaRL (CoRL 2025): simpler reward but still needs speed + progress
      - gym-carla: speed, lateral distance, collision, out of lane

    All values are cast to native Python types (no numpy).
    """
    # Ego position from matrix
    ego_matrix = meta.get("ego_matrix")
    if ego_matrix is not None:
        ego_x = float(ego_matrix[0, 3])
        ego_y = float(ego_matrix[1, 3])
    else:
        pos = meta.get("pos_global", [0.0, 0.0, 0.0])
        ego_x = float(pos[0])
        ego_y = float(pos[1])

    # Navigation command
    commands = meta.get("next_commands_4.0", meta.get("next_commands", [4]))
    command = int(commands[0]) if commands else 4

    # Target point
    target_pts = meta.get(
        "next_gps_target_points_4.0",
        meta.get("next_target_points_4.0", meta.get("next_target_points", [])),
    )
    if target_pts and len(target_pts) > 0:
        tp = target_pts[0]
        x_cmd = float(tp[0])
        y_cmd = float(tp[1])
    else:
        x_cmd = ego_x + 10.0
        y_cmd = ego_y

    # Convert numpy scalars to Python types safely
    def to_float(v, default=0.0):
        if v is None:
            return default
        if hasattr(v, "item"):
            return float(v.item())
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def to_bool(v, default=False):
        if v is None:
            return default
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return default

    def to_str(v, default=""):
        if v is None:
            return default
        return str(v)

    meas = {
        # -- Basic ego state --
        "x": ego_x,
        "y": ego_y,
        "theta": to_float(meta.get("theta")),
        "speed": to_float(meta.get("speed")),
        "steer": to_float(meta.get("steer")),
        "throttle": to_float(meta.get("throttle")),
        "brake": 1.0 if meta.get("brake", False) else 0.0,

        # -- Navigation --
        "command": command,
        "x_command": x_cmd,
        "y_command": y_cmd,

        # -- Speed targets (Roach reward) --
        "speed_limit": to_float(meta.get("speed_limit"), 8.33),
        "target_speed": to_float(meta.get("target_speed"), 8.33),

        # -- Route deviation (Roach reward) --
        "distance_ego_to_route": to_float(meta.get("distance_ego_to_route")),
        "ego_lane_width": to_float(meta.get("ego_lane_width"), 3.5),
        "route_left_length": to_float(meta.get("route_left_length"), 100.0),

        # -- Hazard flags (Roach terminal / penalty) --
        "vehicle_hazard": to_bool(meta.get("vehicle_hazard")),
        "walker_hazard": to_bool(meta.get("walker_hazard")),
        "light_hazard": to_bool(meta.get("light_hazard")),
        "stop_sign_hazard": to_bool(meta.get("stop_sign_hazard")),
        "stop_sign_close": to_bool(meta.get("stop_sign_close")),

        # -- Traffic context --
        "is_junction": to_bool(meta.get("is_junction")),
        "traffic_light_state": to_str(meta.get("traffic_light_state"), "None"),

        # -- Nearby objects --
        "dist_to_pedestrian": to_float(meta.get("dist_to_pedestrian"), 999.0),
        "speed_reduced_by_obj_distance": to_float(
            meta.get("speed_reduced_by_obj_distance"), 999.0
        ),
        "speed_reduced_by_obj_type": to_str(
            meta.get("speed_reduced_by_obj_type"), "none"
        ),

        # -- Lane info --
        "lane_id": int(meta.get("lane_id", 0)) if meta.get("lane_id") is not None else 0,
        "ego_lane_id": int(meta.get("ego_lane_id", 0)) if meta.get("ego_lane_id") is not None else 0,
    }

    return meas


def process_zip(zip_path, output_dir):
    """Process one route zip: extract JPGs and rich measurements."""
    route_name = zip_path.stem
    rgb_out = output_dir / "rgb"
    meas_out = output_dir / "measurements"
    rgb_out.mkdir(parents=True, exist_ok=True)
    meas_out.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        jpg_files = sorted([n for n in names if "/rgb/" in n and n.endswith(".jpg")])
        pkl_files = sorted([n for n in names if "/metas/" in n and n.endswith(".pkl")])

        if not jpg_files or not pkl_files:
            return 0

        for jpg_path, pkl_path in zip(jpg_files, pkl_files):
            frame_id = Path(jpg_path).stem

            # Extract JPG (only if not already there)
            img_out = rgb_out / f"{frame_id}.jpg"
            if not img_out.exists():
                with zf.open(jpg_path) as src:
                    img_out.write_bytes(src.read())

            # Extract and convert pickle to rich JSON
            try:
                with zf.open(pkl_path) as src:
                    raw = src.read()
                    meta = pickle.loads(lzma.decompress(raw))
            except Exception:
                continue

            if not isinstance(meta, dict):
                continue

            meas = extract_rich_measurements(meta)

            with open(meas_out / f"{frame_id}.json", "w") as f:
                json.dump(meas, f)

            frame_count += 1

    return frame_count


def main():
    print("=" * 60)
    print("  Re-extracting LEAD data with rich reward fields")
    print("=" * 60)

    cached_zips = find_cached_zips()
    if not cached_zips:
        print("No cached zips found. Run download_real_data.py first.")
        return

    # Find which routes we already have
    existing_routes = set()
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and (d / "rgb").exists():
                existing_routes.add(d.name)

    print(f"Existing routes: {len(existing_routes)}")

    total_updated = 0
    total_frames = 0

    for i, zip_path in enumerate(cached_zips):
        route_name = zip_path.stem

        # Only process routes we already downloaded
        if route_name not in existing_routes:
            continue

        out_dir = OUTPUT_DIR / route_name
        n = process_zip(zip_path, out_dir)
        total_frames += n
        total_updated += 1

        if (i + 1) % 50 == 0 or total_updated == len(existing_routes):
            print(f"  Processed {total_updated}/{len(existing_routes)} routes, {total_frames} frames")

    print(f"\nDone. Updated {total_updated} routes with {total_frames} frames.")

    # Verify
    sample_dir = next(OUTPUT_DIR.iterdir())
    sample_meas = next((sample_dir / "measurements").glob("*.json"))
    with open(sample_meas) as f:
        m = json.load(f)
    print(f"\nSample measurement fields ({len(m)} total):")
    for k in sorted(m.keys()):
        print(f"  {k}: {repr(m[k])[:80]}")


if __name__ == "__main__":
    main()
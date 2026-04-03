"""
Download real CARLA driving data from the LEAD dataset (CVPR 2026).

This downloads route zips from HuggingFace, extracts RGB images and
metadata pickles, and converts them into TransFuser format that our
dataset.py can load directly.

Each route is ~15-100MB. We download ~100 routes = ~3-5GB total.
This gives us ~10K+ frames — enough for a meaningful BC model.

Usage:
    python scripts/download_real_data.py                  # default: 100 routes
    python scripts/download_real_data.py --max-routes 50  # fewer for testing
    python scripts/download_real_data.py --max-routes 200 # more data
"""

import argparse
import json
import lzma
import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


HF_REPO = "ln2697/lead_carla"
OUTPUT_DIR = Path("data_raw/transfuser")

# Scenario folders in the LEAD dataset
# We pick diverse scenarios for good training coverage
SCENARIOS = [
    "Accident",
    "ControlLoss",
    "CrossingBicycleFlow",
    "DynamicObjectCrossing",
    "EnterActorFlow",
    "HardBreakRoute",
    "HighwayCutIn",
    "HighwayExit",
    "InterurbanActorFlow",
    "InvadingTurn",
    "MergerIntoSlowTraffic",
    "NonSignalizedJunctionLeftTurn",
    "OppositeVehicleTakingPriority",
    "ParkingCrossingPedestrian",
    "PedestrianCrossing",
    "SignalizedJunctionLeftTurn",
    "SignalizedJunctionRightTurn",
    "StaticCutIn",
    "VanillaNonSignalizedTurn",
    "VanillaSignalizedTurnEncounterGreenLight",
    "VehicleTurningRoutePedestrian",
    "YieldToEmergencyVehicle",
]


def install_huggingface_hub():
    """Ensure huggingface_hub is installed."""
    try:
        from huggingface_hub import list_repo_tree, hf_hub_download
        return True
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
        )
        return True


def list_available_routes():
    """List all available route zips from HuggingFace, grouped by town."""
    from huggingface_hub import list_repo_tree

    print("Fetching route list from HuggingFace...")
    all_routes = []

    for scenario in SCENARIOS:
        try:
            files = list(
                list_repo_tree(HF_REPO, repo_type="dataset", path_in_repo=scenario)
            )
            zips = [f.path for f in files if f.path.endswith(".zip")]
            all_routes.extend(zips)
        except Exception:
            # Scenario folder might not exist
            continue

    print(f"  Found {len(all_routes)} routes across {len(SCENARIOS)} scenario types")
    return all_routes


def extract_town(route_path: str) -> str:
    """Extract town name from route path like 'Accident/Town03_Rep0_route_...'."""
    filename = route_path.split("/")[-1]
    # Format: Town03_Rep0_route_001792_...
    parts = filename.split("_")
    return parts[0]  # "Town03"


def convert_route(zip_path: str, route_name: str, output_dir: Path) -> int:
    """
    Extract a route zip and convert to TransFuser format.

    LEAD format: rgb/*.jpg + metas/*.pkl (xz-compressed)
    TransFuser format: rgb/*.png + measurements/*.json

    Returns number of frames converted.
    """
    rgb_out = output_dir / "rgb"
    meas_out = output_dir / "measurements"
    rgb_out.mkdir(parents=True, exist_ok=True)
    meas_out.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Find the nested path prefix
        # e.g., "data/carla_leaderboard2//data/Accident/Town03_.../rgb/0000.jpg"
        rgb_files = sorted([n for n in names if "/rgb/" in n and n.endswith(".jpg")])
        pkl_files = sorted([n for n in names if "/metas/" in n and n.endswith(".pkl")])

        if not rgb_files or not pkl_files:
            return 0

        for jpg_path, pkl_path in zip(rgb_files, pkl_files):
            frame_id = Path(jpg_path).stem  # "0000", "0001", etc.

            # Extract and save RGB image (keep as jpg — our dataset.py will handle it)
            with zf.open(jpg_path) as src:
                img_data = src.read()
            img_out_path = rgb_out / f"{frame_id}.jpg"
            with open(img_out_path, "wb") as dst:
                dst.write(img_data)

            # Extract and convert metadata pickle → JSON
            try:
                with zf.open(pkl_path) as src:
                    raw = src.read()
                    decompressed = lzma.decompress(raw)
                    meta = pickle.loads(decompressed)
            except Exception:
                continue

            if not isinstance(meta, dict):
                continue

            # Extract ego position from ego_matrix
            ego_matrix = meta.get("ego_matrix")
            if ego_matrix is not None:
                ego_x = float(ego_matrix[0, 3])
                ego_y = float(ego_matrix[1, 3])
            else:
                pos = meta.get("pos_global", [0.0, 0.0, 0.0])
                ego_x = float(pos[0])
                ego_y = float(pos[1])

            # Get navigation command and target point
            # next_commands_4.0 gives commands at 4m lookahead distance
            commands = meta.get("next_commands_4.0", meta.get("next_commands", [4]))
            command = int(commands[0]) if commands else 4

            # Target point in world coords
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

            # Build TransFuser-compatible measurement dict
            # All values explicitly native Python types
            meas = {
                "x": ego_x,
                "y": ego_y,
                "theta": float(meta.get("theta", 0.0)),
                "speed": float(meta.get("speed", 0.0)),
                "steer": float(meta.get("steer", 0.0)),
                "throttle": float(meta.get("throttle", 0.0)),
                "brake": 1.0 if meta.get("brake", False) else 0.0,
                "x_command": x_cmd,
                "y_command": y_cmd,
                "command": command,
            }

            with open(meas_out / f"{frame_id}.json", "w") as f:
                json.dump(meas, f)

            frame_count += 1

    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Download real CARLA driving data")
    parser.add_argument(
        "--max-routes", type=int, default=100, help="Max routes to download"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroDriver — Real CARLA Data Download")
    print("  Source: LEAD dataset (CVPR 2026)")
    print("=" * 60)

    install_huggingface_hub()
    from huggingface_hub import hf_hub_download

    # List all available routes
    all_routes = list_available_routes()

    if not all_routes:
        print("ERROR: Could not find any routes. Check your internet connection.")
        sys.exit(1)

    # Group by town
    town_routes = {}
    for route in all_routes:
        town = extract_town(route)
        town_routes.setdefault(town, []).append(route)

    print("\nRoutes per town:")
    for town in sorted(town_routes.keys()):
        print(f"  {town}: {len(town_routes[town])} routes")

    # Select routes: train from Town01-04, val from Town05
    train_towns = ["Town01", "Town02", "Town03", "Town04"]
    val_towns = ["Town05", "Town06"]

    train_routes = []
    for t in train_towns:
        train_routes.extend(town_routes.get(t, []))

    val_routes = []
    for t in val_towns:
        val_routes.extend(town_routes.get(t, []))

    # Cap at max_routes
    max_train = int(args.max_routes * 0.8)
    max_val = args.max_routes - max_train

    train_routes = train_routes[:max_train]
    val_routes = val_routes[:max_val]

    print(f"\nDownload plan:")
    print(f"  Train: {len(train_routes)} routes from {train_towns}")
    print(f"  Val:   {len(val_routes)} routes from {val_towns}")

    est_gb = (len(train_routes) + len(val_routes)) * 0.04
    print(f"  Estimated download: ~{est_gb:.1f} GB")

    free_gb = shutil.disk_usage(Path.home()).free / (1024**3)
    print(f"  Free disk space: {free_gb:.1f} GB")

    if est_gb > free_gb * 0.8:
        print(f"\n⚠ This might be tight on disk. Reduce --max-routes if needed.")

    resp = input(f"\nProceed? [Y/n]: ")
    if resp.lower() == "n":
        sys.exit(0)

    # Delete old dummy data if present
    dummy_dirs = list(OUTPUT_DIR.glob("routes_town*_test_*"))
    dummy_dirs += list(OUTPUT_DIR.glob("routes_town*_long_*"))
    if dummy_dirs:
        print(f"\nRemoving {len(dummy_dirs)} old dummy route(s)...")
        for d in dummy_dirs:
            shutil.rmtree(d, ignore_errors=True)

    # Download and convert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    total_routes = 0

    for label, routes in [("TRAIN", train_routes), ("VAL", val_routes)]:
        print(f"\n--- Downloading {label} routes ---")

        for i, route_path in enumerate(routes):
            route_filename = route_path.split("/")[-1]
            route_name = route_filename.replace(".zip", "")
            out_dir = OUTPUT_DIR / route_name

            # Skip if already converted
            if out_dir.exists() and (out_dir / "rgb").exists():
                existing = len(list((out_dir / "rgb").glob("*.jpg")))
                if existing > 0:
                    total_frames += existing
                    total_routes += 1
                    print(f"  [{i+1}/{len(routes)}] {route_name}: cached ({existing} frames)")
                    continue

            print(f"  [{i+1}/{len(routes)}] {route_name}...", end=" ", flush=True)

            try:
                zip_local = hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=route_path,
                )

                n_frames = convert_route(zip_local, route_name, out_dir)

                if n_frames > 0:
                    total_frames += n_frames
                    total_routes += 1
                    print(f"✓ {n_frames} frames")
                else:
                    print(f"⚠ 0 frames (skipped)")
                    shutil.rmtree(out_dir, ignore_errors=True)

            except Exception as e:
                print(f"✗ Error: {e}")
                shutil.rmtree(out_dir, ignore_errors=True)
                continue

    # Summary
    print(f"\n  ")
    print(f"  Download Complete!")
    print(f"  Routes: {total_routes}")
    print(f"  Frames: {total_frames}")
    print(f"  Location: {OUTPUT_DIR}")
    print(f"  ")

    # Show data size
    total_bytes = sum(
        f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()
    )
    print(f"  Disk usage: {total_bytes / 1e9:.2f} GB")

    print(f"\n  Next steps:")
    print(f"  1. python -m neurodriver.data.dataset    # verify data loads")
    print(f"  2. python -m neurodriver.training.train_bc --config configs/bc.yaml")


if __name__ == "__main__":
    main()
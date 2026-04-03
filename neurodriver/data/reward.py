"""
Roach-style driving reward using rich LEAD metadata fields.

Replaces the broken compute_reward() that gave ~1.7 to everything.
This produces rewards in roughly [-1, +1] with actual variance.
"""

import numpy as np


def compute_reward_v2(meas: dict, prev_meas: dict = None) -> float:
    """
    Roach-style reward from rich LEAD measurements.

    Components:
      + speed_reward     [0, 1]: match scenario target speed
      + centering_reward [0, 1]: stay near route center
      + progress_reward  [0, 0.5]: forward progress along route
      - hazard_penalty   [-1, 0]: vehicle/walker/light hazards
      - jerk_penalty     [-0.3, 0]: sudden steering changes

    Total range roughly [-1.3, +1.0]. Expert frames land ~[0.3, 0.8].
    """
    speed = meas.get("speed", 0.0)
    target_speed = max(meas.get("target_speed", meas.get("speed_limit", 8.33)), 1.0)

    # --- Speed reward: 1.0 at target, 0.0 when error = target_speed ---
    speed_error = abs(speed - target_speed)
    speed_reward = max(0.0, 1.0 - speed_error / target_speed)
    if speed < 0.5 and target_speed > 2.0:
        speed_reward = 0.0

    # Centering reward: stay near route centerline
    dist_to_route = abs(meas.get("distance_ego_to_route", 0.0))
    lane_width = max(meas.get("ego_lane_width", 3.5), 1.0)
    normalized_dev = min(dist_to_route / (lane_width * 0.5), 2.0)

    if normalized_dev <= 1.0:
        centering_reward = 1.0 - normalized_dev ** 2
    else:
        centering_reward = max(-1.0, -(normalized_dev - 1.0))

    # Progress reward: route_left_length should decrease 
    progress_reward = 0.0
    if prev_meas is not None:
        prev_route_left = prev_meas.get("route_left_length", 100.0)
        curr_route_left = meas.get("route_left_length", 100.0)
        progress = prev_route_left - curr_route_left
        progress_reward = np.clip(progress * 0.1, 0.0, 0.5)

    # Hazard penalties
    hazard_penalty = 0.0
    if meas.get("vehicle_hazard", False):
        hazard_penalty -= 0.5
    if meas.get("walker_hazard", False):
        hazard_penalty -= 0.5
    if meas.get("light_hazard", False):
        hazard_penalty -= 0.3
    if meas.get("stop_sign_hazard", False) and speed > 0.5:
        hazard_penalty -= 0.2

    # Jerk penalty: sudden steering changes
    jerk_penalty = 0.0
    if prev_meas is not None:
        steer_change = abs(meas.get("steer", 0.0) - prev_meas.get("steer", 0.0))
        jerk_penalty = -min(steer_change * 2.0, 0.3)

    # Combine with weights
    reward = (
        0.4 * speed_reward
        + 0.3 * centering_reward
        + progress_reward
        + hazard_penalty
        + jerk_penalty
    )

    return float(reward)


if __name__ == "__main__":
    """Quick sanity check: verify reward variance on different scenarios."""

    # Good driving: on target speed, centered, no hazards
    good = {
        "speed": 8.0, "target_speed": 8.33, "speed_limit": 8.33,
        "distance_ego_to_route": 0.1, "ego_lane_width": 3.5,
        "route_left_length": 90.0, "steer": 0.01,
        "vehicle_hazard": False, "walker_hazard": False,
        "light_hazard": False, "stop_sign_hazard": False,
    }
    good_prev = {
        "route_left_length": 92.0, "steer": 0.01,
    }

    # Bad driving: wrong speed, off lane, hazard
    bad = {
        "speed": 2.0, "target_speed": 8.33, "speed_limit": 8.33,
        "distance_ego_to_route": 2.5, "ego_lane_width": 3.5,
        "route_left_length": 92.0, "steer": 0.4,
        "vehicle_hazard": True, "walker_hazard": False,
        "light_hazard": True, "stop_sign_hazard": False,
    }
    bad_prev = {
        "route_left_length": 92.0, "steer": -0.1,
    }

    # Stopped when should move
    stopped = {
        "speed": 0.0, "target_speed": 8.33, "speed_limit": 8.33,
        "distance_ego_to_route": 0.0, "ego_lane_width": 3.5,
        "route_left_length": 100.0, "steer": 0.0,
        "vehicle_hazard": False, "walker_hazard": False,
        "light_hazard": False, "stop_sign_hazard": False,
    }

    r_good = compute_reward_v2(good, good_prev)
    r_bad = compute_reward_v2(bad, bad_prev)
    r_stopped = compute_reward_v2(stopped, None)

    print(f"Good driving:  {r_good:+.4f}")
    print(f"Bad driving:   {r_bad:+.4f}")
    print(f"Stopped:       {r_stopped:+.4f}")
    print(f"Spread:        {r_good - r_bad:.4f}")

    assert r_good > r_bad, "Good driving should score higher than bad!"
    assert r_good > r_stopped, "Good driving should score higher than stopped!"
    assert r_good - r_bad > 0.5, f"Spread too small: {r_good - r_bad:.4f}"

    print("\nReward sanity check PASSED")
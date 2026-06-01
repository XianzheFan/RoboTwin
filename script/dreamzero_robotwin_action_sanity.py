#!/usr/bin/env python3
"""Sanity-check RoboTwin demos against DreamZero action metadata.

This is intentionally offline and read-only. It verifies the flat qpos/action
layout, recomputes the DreamZero relative joint statistics from RoboTwin HDF5
demos, optionally compares those stats against a checkpoint metadata.json and a
LeRobot root, and reports whether requested eval seeds are part of the demo set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ARM_SLICES = {
    "panda0_joint_pos": slice(0, 7),
    "panda1_joint_pos": slice(8, 15),
}


def _round_list(values: np.ndarray, ndigits: int = 5) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metadata_path(ckpt: Path) -> Path:
    candidates = [
        ckpt / "experiment_cfg" / "metadata.json",
        ckpt / "metadata.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"metadata.json not found under {ckpt}")


def _episode_id(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits)


def _load_hdf5_qpos(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        left_arm = f["joint_action/left_arm"][()].astype(np.float32)
        left_gripper = f["joint_action/left_gripper"][()].astype(np.float32)
        right_arm = f["joint_action/right_arm"][()].astype(np.float32)
        right_gripper = f["joint_action/right_gripper"][()].astype(np.float32)
        vector = f["joint_action/vector"][()].astype(np.float32)

    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]

    rebuilt = np.concatenate(
        [left_arm, left_gripper, right_arm, right_gripper],
        axis=1,
    )
    if vector.shape != rebuilt.shape:
        raise ValueError(f"{path}: vector shape {vector.shape} != rebuilt {rebuilt.shape}")
    max_err = float(np.max(np.abs(vector - rebuilt))) if len(vector) else 0.0
    if max_err > 1e-5:
        raise ValueError(f"{path}: joint_action/vector layout mismatch max_err={max_err}")
    return vector


def _compute_offsets(
    episode_paths: Iterable[Path],
    action_horizon: int,
    step_filter: dict[int, set[int]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    per_step: list[np.ndarray] = []
    per_chunk: list[np.ndarray] = []
    lengths: list[int] = []

    for converted_ep_idx, path in enumerate(episode_paths):
        qpos = _load_hdf5_qpos(path)
        lengths.append(len(qpos))
        if len(qpos) < 2:
            continue
        per_step.append(qpos[1:] - qpos[:-1])

        # Converter stores action[t] = state[t + 1]. The DreamZero loader
        # samples action offsets [0..H-1] and subtracts state at the chunk
        # anchor, so the physical offsets are qpos[i+1:i+H+1] - qpos[i].
        filtered_indices = (step_filter or {}).get(converted_ep_idx, set())
        for start in range(max(0, len(qpos) - action_horizon)):
            if start in filtered_indices:
                continue
            offsets = qpos[start + 1 : start + action_horizon + 1] - qpos[start]
            if offsets.shape[0] == action_horizon:
                per_chunk.append(offsets)

    step_arr = np.concatenate(per_step, axis=0) if per_step else np.empty((0, 16))
    chunk_arr = np.concatenate(per_chunk, axis=0) if per_chunk else np.empty((0, 16))
    return (
        {key: step_arr[:, slc] for key, slc in ARM_SLICES.items()},
        {key: chunk_arr[:, slc] for key, slc in ARM_SLICES.items()},
        lengths,
    )


def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        raise ValueError("cannot compute stats for empty array")
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
        "abs_p50": np.quantile(np.abs(arr), 0.50, axis=0),
        "abs_max": np.max(np.abs(arr), axis=0),
    }


def _stats_for_json(stats: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: _round_list(value) for key, value in stats.items()}


def _max_abs_diff(lhs: list[float] | np.ndarray, rhs: list[float] | np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64))))


def _load_checkpoint_action_stats(ckpt: Path | None) -> dict:
    if ckpt is None:
        return {}
    metadata = _read_json(_metadata_path(ckpt))
    return (
        metadata.get("robofactory", {})
        .get("statistics", {})
        .get("action", {})
    )


def _load_lerobot_relative_stats(root: Path | None) -> dict:
    if root is None:
        return {}
    path = root / "meta" / "relative_stats_dreamzero.json"
    return _read_json(path) if path.exists() else {}


def _load_lerobot_step_filter(root: Path | None) -> dict[int, set[int]]:
    if root is None:
        return {}
    path = root / "meta" / "step_filter.jsonl"
    if not path.exists():
        return {}
    result: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            result[int(item["episode_index"])] = {
                int(index) for index in item.get("step_indices", [])
            }
    return result


def _read_seeds(demo_root: Path) -> list[int]:
    path = demo_root / "seed.txt"
    if not path.exists():
        return []
    return [int(x) for x in path.read_text(encoding="utf-8").split()]


def _count_sides(demo_root: Path) -> dict[str, int]:
    path = demo_root / "scene_info.json"
    if not path.exists():
        return {}
    scene = _read_json(path)
    counts: dict[str, int] = {}
    for item in scene.values():
        side = item.get("info", {}).get("{a}") if isinstance(item, dict) else None
        if side:
            counts[side] = counts.get(side, 0) + 1
    return counts


def _verify_lerobot_first_episode(lerobot_root: Path | None, demo_root: Path) -> dict:
    if lerobot_root is None:
        return {"checked": False}
    parquet = lerobot_root / "data" / "chunk-000" / "episode_000000.parquet"
    if not parquet.exists():
        return {"checked": False, "reason": f"{parquet} missing"}
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional environment dep
        return {"checked": False, "reason": f"pyarrow unavailable: {exc}"}

    table = pq.read_table(parquet, columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    h5_qpos = _load_hdf5_qpos(demo_root / "data" / "episode0.hdf5")
    state_n = min(len(state), len(h5_qpos) - 1)
    action_n = min(len(action), len(h5_qpos) - 1)
    state_err = _max_abs_diff(state[:state_n], h5_qpos[:state_n])
    action_err = _max_abs_diff(action[:action_n], h5_qpos[1 : action_n + 1])
    return {
        "checked": True,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "state_vs_hdf5_max_abs_err": state_err,
        "action_vs_hdf5_next_qpos_max_abs_err": action_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", required=True, type=Path)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    demo_root = args.demo_root
    episode_paths = sorted((demo_root / "data").glob("episode*.hdf5"), key=_episode_id)
    if not episode_paths:
        raise FileNotFoundError(f"no episode*.hdf5 files under {demo_root / 'data'}")

    step_filter = _load_lerobot_step_filter(args.lerobot_root)
    step_offsets, chunk_offsets, lengths = _compute_offsets(
        episode_paths,
        action_horizon=args.action_horizon,
        step_filter=step_filter,
    )
    chunk_stats = {key: _stats(value) for key, value in chunk_offsets.items()}
    step_stats = {key: _stats(value) for key, value in step_offsets.items()}

    seeds = _read_seeds(demo_root)
    eval_seeds = [
        int(item)
        for item in args.eval_seeds.replace(",", " ").split()
        if item.strip()
    ]
    seed_membership = {str(seed): seed in seeds for seed in eval_seeds}

    checkpoint_stats = _load_checkpoint_action_stats(args.ckpt)
    lerobot_relative_stats = _load_lerobot_relative_stats(args.lerobot_root)
    comparisons: dict[str, dict[str, float]] = {}
    for key in ARM_SLICES:
        comparisons[key] = {}
        if key in checkpoint_stats:
            comparisons[key]["ckpt_q01_max_abs_diff"] = _max_abs_diff(
                checkpoint_stats[key]["q01"], chunk_stats[key]["q01"]
            )
            comparisons[key]["ckpt_q99_max_abs_diff"] = _max_abs_diff(
                checkpoint_stats[key]["q99"], chunk_stats[key]["q99"]
            )
        if key in lerobot_relative_stats:
            comparisons[key]["lerobot_q01_max_abs_diff"] = _max_abs_diff(
                lerobot_relative_stats[key]["q01"], chunk_stats[key]["q01"]
            )
            comparisons[key]["lerobot_q99_max_abs_diff"] = _max_abs_diff(
                lerobot_relative_stats[key]["q99"], chunk_stats[key]["q99"]
            )

    result = {
        "demo_root": str(demo_root),
        "num_episodes": len(episode_paths),
        "seed_count": len(seeds),
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "eval_seed_membership": seed_membership,
        "side_counts": _count_sides(demo_root),
        "step_filter": {
            "source": str(args.lerobot_root / "meta" / "step_filter.jsonl")
            if step_filter and args.lerobot_root is not None
            else None,
            "episodes": len(step_filter),
            "filtered_indices": sum(len(indices) for indices in step_filter.values()),
        },
        "trajectory_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(float(np.mean(lengths)), 3),
        },
        "step_delta_stats": {
            key: _stats_for_json(step_stats[key]) for key in ARM_SLICES
        },
        "chunk_relative_stats": {
            key: _stats_for_json(chunk_stats[key]) for key in ARM_SLICES
        },
        "metadata_comparison": comparisons,
        "lerobot_episode0_check": _verify_lerobot_first_episode(
            args.lerobot_root,
            demo_root,
        ),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    bad_diffs = [
        value
        for comparison in comparisons.values()
        for value in comparison.values()
        if not math.isfinite(value) or value > 1e-4
    ]
    lerobot_check = result["lerobot_episode0_check"]
    if lerobot_check.get("checked"):
        bad_diffs.extend(
            value
            for key, value in lerobot_check.items()
            if key.endswith("_max_abs_err") and value > 1e-5
        )
    if bad_diffs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

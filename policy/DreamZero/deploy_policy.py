"""RoboTwin entry-point for the DreamZero multi-agent (bimanual) policy.

Each arm is treated as an independent agent (left = agent 0,
right = agent 1). This adapter is intentionally lightweight so it can
run inside the egl+1.0 container (Ubuntu 18.04, glibc 2.27, Python 3.9)
where the dreamzero VLA itself cannot import (it needs Python 3.11 and
torch 2.8). All heavy lifting happens in a separate process:

  +-----------------+ ws (msgpack-numpy) +-----------------------+
  | RoboTwin sim    | -------------------> | bimanual_policy_server |
  | (this file)     | <------------------- | dreamzero VLA + LoRA  |
  +-----------------+                      +-----------------------+

The server (``dreamzero/eval_utils/bimanual_policy_server.py``) holds the
rolling 33-frame history and the inverse-q99 normalization, so this
client just sends per-step obs and applies the returned absolute-qpos
actions.

Wire protocol: see ``bimanual_policy_server.py`` docstring.
"""

from __future__ import annotations

import os
import socket
import time
import uuid

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# msgpack-numpy compat: prefer the openpi-style API (matches the
# single-arm WebsocketClientPolicy in dreamzero/eval_utils) but fall back
# to the upstream msgpack-numpy package when running inside the
# container's env. Both serialize numpy arrays identically.
# ---------------------------------------------------------------------------
try:
    from openpi_client import msgpack_numpy as _mn  # type: ignore

    _PACK = _mn.Packer().pack
    _UNPACK = _mn.unpackb
except Exception:  # pragma: no cover — fallback path
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()

    def _PACK(obj):  # noqa: N802
        return msgpack.packb(obj, use_bin_type=True)

    def _UNPACK(buf):  # noqa: N802
        return msgpack.unpackb(buf, raw=False)


def _connect(uri: str, timeout_s: float = 120.0, retry_every: float = 2.0):
    """Connect to the bimanual server, retrying for ``timeout_s`` so the
    SLURM job can wait for the server-side job to finish loading
    the 14B Wan backbone (multi-minute warmup on H100).
    """
    import websockets.sync.client as wsc  # local import: only avail in env

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            return wsc.connect(
                uri,
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=600,
            )
        except (ConnectionRefusedError, OSError, socket.gaierror) as e:
            last_err = e
            time.sleep(retry_every)
    raise RuntimeError(
        f"Failed to connect to bimanual server at {uri} within "
        f"{timeout_s}s: {last_err}"
    )


def encode_obs(observation):
    """Pick head/left/right rgb + 16-dim qpos out of a RoboTwin obs dict.

    RoboTwin's ``TASK_ENV.get_obs()`` returns a nested dict::

        {
            "observation": {
                "head_camera":  {"rgb": [H, W, 3] uint8, ...},
                "left_camera":  {"rgb": [H, W, 3] uint8, ...},
                "right_camera": {"rgb": [H, W, 3] uint8, ...},
            },
            "joint_action": {
                "vector":       [16] float,
                "left_arm":     [7] float, "left_gripper":  scalar,
                "right_arm":    [7] float, "right_gripper": scalar,
            },
            ...
        }
    """
    obs_root = observation.get("observation", observation)
    head_rgb = np.asarray(obs_root["head_camera"]["rgb"], dtype=np.uint8)
    left_rgb = np.asarray(obs_root["left_camera"]["rgb"], dtype=np.uint8)
    right_rgb = np.asarray(obs_root["right_camera"]["rgb"], dtype=np.uint8)

    joint = observation.get("joint_action", {})
    if "vector" in joint:
        qpos = np.asarray(joint["vector"], dtype=np.float32)
    else:
        qpos = np.concatenate([
            np.asarray(joint["left_arm"], dtype=np.float32).ravel(),
            np.asarray(joint["left_gripper"], dtype=np.float32).ravel(),
            np.asarray(joint["right_arm"], dtype=np.float32).ravel(),
            np.asarray(joint["right_gripper"], dtype=np.float32).ravel(),
        ])
    assert qpos.shape == (16,), f"expected 16-dim qpos, got {qpos.shape}"

    return {
        "head_rgb": head_rgb,
        "left_rgb": left_rgb,
        "right_rgb": right_rgb,
        "qpos": qpos,
    }


class DreamZeroBimanualPolicy:
    """RoboTwin policy adapter that proxies to a bimanual WebSocket server."""

    def __init__(self, usr_args):
        self.server_host = usr_args.get(
            "server_host", os.environ.get("DREAMZERO_HOST", "127.0.0.1")
        )
        self.server_port = int(usr_args.get(
            "server_port", os.environ.get("DREAMZERO_PORT", "5001")
        ))
        self.replan_every = int(usr_args.get("replan_every", 8))
        self.connect_timeout = float(usr_args.get("connect_timeout", 600.0))
        self.fallback_instruction = usr_args.get(
            "fallback_instruction", "use the two robot arms to complete the task"
        )

        # Per-episode session id so the server can keep its rolling
        # video history scoped to one rollout.
        self.session_id: str = uuid.uuid4().hex
        self.obs_cache = []  # only stores latest for RoboTwin's empty-check
        self._pending_actions: list[np.ndarray] = []
        self._ws = None
        self._server_meta: dict | None = None
        self._instruction: str = self.fallback_instruction
        self._last_qpos: np.ndarray | None = None
        self._needs_reset: bool = True

    # ---- public hooks RoboTwin expects --------------------------------
    def set_instruction(self, text: str) -> None:
        self._instruction = text or self.fallback_instruction

    def update_obs(self, obs: dict) -> None:
        # Stash only the latest; the server keeps the rolling history.
        self.obs_cache = [obs]
        self._last_qpos = obs["qpos"]

    def reset(self) -> None:
        self.obs_cache = []
        self._pending_actions.clear()
        self._last_qpos = None
        self._instruction = self.fallback_instruction
        # New episode -> new session id, then ask the server to clear
        # its history for that session.
        self.session_id = uuid.uuid4().hex
        self._needs_reset = True

    # ---- connection management ----------------------------------------
    def _ensure_connected(self):
        if self._ws is not None:
            return
        uri = f"ws://{self.server_host}:{self.server_port}"
        self._ws = _connect(uri, timeout_s=self.connect_timeout)
        self._server_meta = _UNPACK(self._ws.recv())
        # Light validation so a server/client schema drift surfaces early.
        assert self._server_meta.get("num_agents") == 2, (
            f"Server reports num_agents={self._server_meta.get('num_agents')}, "
            "expected 2 (bimanual)"
        )

    def _send(self, payload: dict) -> dict:
        self._ensure_connected()
        self._ws.send(_PACK(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Bimanual server error:\n{response}")
        return _UNPACK(response)

    def _reset_server_if_needed(self) -> None:
        if not self._needs_reset:
            return
        self._send({
            "endpoint": "reset",
            "session_id": self.session_id,
            "prompt": self._instruction,
        })
        self._needs_reset = False

    # ---- main action chunk request ------------------------------------
    def get_action(self) -> list[np.ndarray]:
        if self._pending_actions:
            actions, self._pending_actions = self._pending_actions, []
            return actions
        if not self.obs_cache:
            raise RuntimeError("No observation cached; call update_obs() first")

        self._reset_server_if_needed()

        obs = self.obs_cache[-1]
        reply = self._send({
            "endpoint": "infer",
            "session_id": self.session_id,
            "qpos": obs["qpos"].astype(np.float32),
            "head_rgb": obs["head_rgb"],
            "left_rgb": obs["left_rgb"],
            "right_rgb": obs["right_rgb"],
            "prompt": self._instruction,
        })
        flat_action = np.asarray(reply["action_chunk"], dtype=np.float32)
        if flat_action.ndim != 2 or flat_action.shape[1] != 16:
            raise RuntimeError(
                f"Unexpected action_chunk shape {flat_action.shape}; "
                "want [T_a, 16]"
            )

        # The RoboTwin/DreamZero training config uses relative_action=false,
        # so the server inverse-normalizes directly into absolute qpos
        # targets. Feed those targets through unchanged to action_type='qpos'.
        absolute_actions = [
            action.astype(np.float32, copy=True) for action in flat_action
        ]

        to_run = absolute_actions[: self.replan_every]
        self._pending_actions = absolute_actions[self.replan_every :]
        return to_run


def get_model(usr_args):
    return DreamZeroBimanualPolicy(usr_args)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    if len(model.obs_cache) == 0:
        model.update_obs(obs)

    instruction = TASK_ENV.get_instruction()
    if instruction:
        model.set_instruction(instruction)

    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        next_obs = TASK_ENV.get_obs()
        model.update_obs(encode_obs(next_obs))


def reset_model(model):
    model.reset()

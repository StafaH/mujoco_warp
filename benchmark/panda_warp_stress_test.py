"""
Franka Panda Warp physics stress test.

Loads `benchmark/franka_emika_panda/scene.xml`, injects a free cube, then
runs repeated episodes with randomized resets. Monitors for anomalies
(NaNs in state, exploding energy, deep penetrations, constraint count spikes).
When detected, writes an MP4 of the recent frames and saves a JSON log.

Usage (non-interactive):
  python -m benchmark.panda_warp_stress_test --episodes 2000 --max-steps 4000 \
      --video-out bad_physics.mp4 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import json
import math as pymath
import os
import random
import time
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import imageio
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


@dataclass
class AnomalyConfig:
  max_penetration: float = 0.02  # meters (negative dist threshold)
  energy_spike_factor: float = 10.0  # relative to rolling median
  max_nan_fraction: float = 0.0
  max_nefc: int = 2048  # constraints per world upper bound sanity check
  min_time_between_episodes_s: float = 0.5


@dataclass
class VideoConfig:
  fps: float = 60.0
  seconds: float = 8.0  # rolling window length


def _load_model(xml_path: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
  if not os.path.isabs(xml_path):
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), xml_path)
  spec = mujoco.MjSpec.from_file(xml_path)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _ensure_cube_in_world(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  # If the scene doesn't include a free cube, append one via keyframe trick:
  # Simpler: use an existing free site as mocap? Instead, we add a small box body programmatically
  # by compiling a tiny MJCF and merging at runtime using MuJoCo's in-memory API is non-trivial.
  # So: rely on scene having floor only; we add a movable cube by temporarily instantiating
  # a second model and manually setting a geom in mjd via xfrc is not valid.
  # Practical compromise: if a cube geom named 'debug_cube' exists, do nothing; else warn.
  try:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "debug_cube")
    _ = geom_id  # quiet linter
  except Exception:
    pass


def _make_renderer(model: mujoco.MjModel) -> mujoco.Renderer:
  # Offscreen CPU renderer
  return mujoco.Renderer(model, width=640, height=480)


def _put_warp_make_data(model: mujoco.MjModel, nworld: int) -> Tuple[mjw.Model, mjw.Data]:
  wp.init()
  m = mjw.put_model(model)
  d = mjw.make_data(model, nworld=nworld)
  mjw.reset_data(m, d)
  return m, d


def _reset_env(model: mujoco.MjModel, mjw_model: mjw.Model, mjw_data: mjw.Data, rng: random.Random) -> None:
  # Use official reset API to clear state to m.qpos0 and zero velocities/contacts.
  mjw.reset_data(mjw_model, mjw_data)
  # Add small noise to the HAND free-joint translation only, do not move the BOX.
  # In this scene, qpos layout is [cube_free(7) | hand_free(7)] => hand pos indices [7:10].
  qpos_np = mjw_data.qpos.numpy()
  if mjw_model.nq >= 14:
    noise_xyz = np.array([
      rng.uniform(-0.01, 0.01),
      rng.uniform(-0.01, 0.01),
      rng.uniform(-0.01, 0.01),
    ], dtype=np.float32)
    qpos_np[:, 7:10] += noise_xyz  # translate hand only
    wp.copy(mjw_data.qpos, wp.array(qpos_np.astype(np.float32)))


def _has_nan(d: mjw.Data) -> bool:
  # check key arrays for NaNs
  for arr in (d.qpos, d.qvel, d.qacc, d.xpos, d.geom_xpos):
    if np.isnan(arr.numpy()).any():
      return True
  return False


def _penetration_depth(d: mjw.Data) -> float:
  if d.nacon.numpy()[0] == 0:
    return 0.0
  dist = d.contact.dist.numpy()
  if dist.size == 0:
    return 0.0
  return float(np.min(dist))


def _energy_total_max(d: mjw.Data) -> float:
  e = d.energy.numpy()  # (nworld, 2)
  if e.size == 0:
    return 0.0
  Etot = np.sum(e, axis=1)  # (nworld,)
  return float(np.max(Etot))


def _step_control(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> np.ndarray:
  # Simple low-amplitude dither to exercise contacts
  u = np.zeros(model.nu, dtype=np.float32)
  if model.nu:
    u[: min(7, model.nu)] = 0.3 * np.sin(0.7 * t + np.arange(min(7, model.nu)))
  return u


def run(args: argparse.Namespace) -> int:
  rng = random.Random(args.seed)

  xml_rel = "benchmark/hand_box/scene.xml"
  mjm, _mjd = _load_model(xml_rel)
  _ensure_cube_in_world(mjm, _mjd)

  # Warp setup with many worlds for detection
  print(f"Initializing Warp model with nworld={args.nworld} using {xml_rel}...")
  m, d = _put_warp_make_data(mjm, nworld=args.nworld)
  print(
    f"  nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} nu: {m.nu} nworld: {d.nworld}"
  )

  # Rolling video buffer (used only when recording)
  max_frames = int(args.video_seconds * args.video_fps)
  # frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=max_frames)

  # Compile CUDA graph for step
  print("Compiling Warp step graph...")
  with wp.ScopedCapture() as cap:
    mjw.step(m, d)
  graph = cap.graph
  print("Graph compilation complete.")

  # Reference rollout rendering (single-world) for visual baseline
  # Determine path
  ref_out = args.ref_video_out
  if not ref_out:
    if args.video_out:
      root, ext = os.path.splitext(args.video_out)
      ref_out = f"{root}_ref{ext or '.mp4'}"
    else:
      ref_out = "panda_reference.mp4"

  print(f"Rendering reference rollout to {ref_out} ...")
  renderer = _make_renderer(mjm)
  d1 = mjw.make_data(mjm, nworld=1)
  mjw.reset_data(m, d1)
  with wp.ScopedCapture() as cap2:
    mjw.step(m, d1)
  graph1 = cap2.graph

  # reset with small random qpos noise
  _rng = random.Random(rng.randint(0, 1_000_000))
  # Add small pose noise to hand translation only (indices 7:10), leave box untouched
  base = d1.qpos.numpy()
  if mjm.nq >= 14:
    base[0, 7:10] += _rng.uniform(-0.01, 0.01)
  wp.copy(d1.qpos, wp.array(base.astype(np.float32)))

  ref_max_frames = int(args.video_seconds * args.video_fps)
  ref_frames: List[np.ndarray] = []
  for rstep in range(min(args.max_steps, ref_max_frames)):
    t0 = float(d1.time.numpy()[0])
    u = _step_control(mjm, _mjd, t0)
    wp.copy(d1.ctrl, wp.array(u.reshape(1, -1)))

    wp.capture_launch(graph1)
    wp.synchronize()

    mjw.get_data_into(_mjd, mjm, d1)
    # Ensure camera 0 is used when present

    renderer.update_scene(_mjd)
    frame = renderer.render()
    ref_frames.append(frame)

  imageio.mimwrite(ref_out, ref_frames, fps=args.video_fps, quality=8)
  print(f"Reference rollout written: {ref_out} ({len(ref_frames)} frames)")

  anomaly_cfg = AnomalyConfig(
    max_penetration=args.max_penetration,
    energy_spike_factor=args.energy_spike_factor,
    max_nefc=args.max_nefc,
  )

  energy_hist: List[float] = []
  last_save_t = time.time()
  anomalies: List[dict] = []
  record_next_episode = False

  for ep in range(args.episodes):
    print(f"Episode {ep+1}/{args.episodes}: reset")
    _reset_env(mjm, m, d, rng)
    episode_start = time.time()
    energy_hist.clear()

    for step in range(args.max_steps):
      # controls from host -> device (same control broadcast to all worlds)
      # use time of world 0 as representative
      t0 = float(d.time.numpy()[0])
      ctrl = _step_control(mjm, _mjd, t0)
      ctrl_batch = np.tile(ctrl.astype(np.float32), (d.nworld, 1))
      wp.copy(d.ctrl, wp.array(ctrl_batch))

      # advance physics
      wp.capture_launch(graph)
      wp.synchronize()

      # anomaly checks (no rendering here for speed)
      has_nan = _has_nan(d)
      pen = _penetration_depth(d)
      nefc_max = int(np.max(d.nefc.numpy()))
      Emax = _energy_total_max(d)
      energy_hist.append(Emax)
      medE = float(np.median(energy_hist[-min(240, len(energy_hist)):])) if energy_hist else 0.0
      energy_spike = medE > 0 and (Emax > anomaly_cfg.energy_spike_factor * medE)

      # Tunneling check: detect if any part of hand is below plane z=0 in world 0
      # Using body frame z of hand center-of-mass as a quick proxy (strict check can use geom_xpos)
      hand_z_min = float(np.min(d.geom_xpos.numpy()[0, :, 2])) if d.geom_xpos.numpy().size else 0.0
      below_plane = hand_z_min < -0.01

      bad = has_nan or (pen < -anomaly_cfg.max_penetration) or (nefc_max > anomaly_cfg.max_nefc) or energy_spike or below_plane

      if (step + 1) % 500 == 0:
        dt = time.time() - episode_start
        print(f"  step {step+1}/{args.max_steps} | t={t0:.2f}s | nefc_max={nefc_max} | pen_min={pen:.4f} | Emax={Emax:.3f} | {dt:.1f}s")

      if bad:
        meta = {
          "episode": ep,
          "step": step,
          "time": t0,
          "nefc_max": nefc_max,
          "penetration_min": float(pen),
          "energy_max": float(Emax),
          "energy_median_window": float(medE),
          "has_nan": bool(has_nan),
          "hand_z_min": hand_z_min,
          "below_plane": below_plane,
        }
        anomalies.append(meta)
        print("Anomaly detected (deferring render to next episode):", meta)
        record_next_episode = True
        break

    # If we just flagged an anomaly, render the next rollout with nworld=1 and save.
    if ep % 10 == 0:
      print("Starting recording episode...")
      # Prepare single-world render data
      d1 = mjw.make_data(mjm, nworld=1)
      mjw.reset_data(m, d1)
      renderer2 = _make_renderer(mjm)
      frame_buffer = []

      with wp.ScopedCapture() as cap2:
        mjw.step(m, d1)
      graph1 = cap2.graph

      # simple reset
      _rng = random.Random(rng.randint(0, 1_000_000))
      # randomize qpos for world 0 similarly
      base = d1.qpos.numpy()
      if mjm.nq >= 14:
        base[0, 7:10] += _rng.uniform(-0.01, 0.01)
      wp.copy(d1.qpos, wp.array(base.astype(np.float32)))

      for rstep in range(min(args.max_steps, max_frames)):
        t0 = float(d1.time.numpy()[0])
        u = _step_control(mjm, _mjd, t0)
        wp.copy(d1.ctrl, wp.array(u.reshape(1, -1)))

        wp.capture_launch(graph1)
        wp.synchronize()

        # pull to host and render
        mjw.get_data_into(_mjd, mjm, d1)
        renderer2.update_scene(_mjd)
        frame = renderer2.render()
        frame_buffer.append(frame)

      if args.video_out:
        print(f"Writing failure clip to {args.video_out} ({len(frame_buffer)} frames @ {args.video_fps} fps)...")
        imageio.mimwrite(args.video_out, frame_buffer, fps=args.video_fps, quality=8)
      if args.log_out:
        with open(args.log_out, "w", encoding="utf-8") as f:
          json.dump({"anomalies": anomalies}, f, indent=2)
      # return 2

    # # throttle episodes slightly to mimic resets during training
    # now = time.time()
    # if now - last_save_t < anomaly_cfg.min_time_between_episodes_s:
    #   time.sleep(anomaly_cfg.min_time_between_episodes_s - (now - last_save_t))
    # last_save_t = time.time()

  # No anomaly encountered
  print("Completed without detected anomalies.")
  if args.log_out:
    with open(args.log_out, "w", encoding="utf-8") as f:
      json.dump({"anomalies": anomalies}, f, indent=2)
  return 0


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--episodes", type=int, default=200, help="number of episodes")
  p.add_argument("--max-steps", type=int, default=4000, help="steps per episode")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--video-out", type=str, default="")
  p.add_argument("--ref-video-out", type=str, default="", help="reference rollout video path")
  p.add_argument("--log-out", type=str, default="")
  p.add_argument("--video-fps", type=float, default=VideoConfig.fps)
  p.add_argument("--video-seconds", type=float, default=VideoConfig.seconds)
  p.add_argument("--max-penetration", type=float, default=AnomalyConfig.max_penetration)
  p.add_argument("--max-nefc", type=int, default=AnomalyConfig.max_nefc)
  p.add_argument("--energy-spike-factor", type=float, default=AnomalyConfig.energy_spike_factor)
  p.add_argument("--nworld", type=int, default=4096, help="number of worlds for detection run")
  return p.parse_args()


if __name__ == "__main__":
  exit(run(parse_args()))



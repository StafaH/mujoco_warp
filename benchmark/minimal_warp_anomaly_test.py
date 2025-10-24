from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import mujoco
import imageio
import numpy as np
import warp as wp

import mujoco_warp as mjw


def load_model(xml_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
  if not os.path.isabs(xml_path):
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), xml_path)
  spec = mujoco.MjSpec.from_file(xml_path)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _world_nan_mask(a: np.ndarray) -> np.ndarray:
  if a.ndim == 1:
    return np.isnan(a)
  return np.isnan(a.reshape(a.shape[0], -1)).any(axis=1)


def find_nan_world(d: mjw.Data) -> Optional[int]:
  masks = []
  for arr in (d.qpos, d.qvel, d.qacc, d.geom_xpos):
    masks.append(_world_nan_mask(arr.numpy()))
  world_mask = np.logical_or.reduce(masks)
  idx = np.flatnonzero(world_mask)
  return int(idx[0]) if idx.size else None


def min_penetration_by_world(d: mjw.Data) -> np.ndarray:
  nworld = d.nworld
  nacon = int(d.nacon.numpy()[0])
  if nacon <= 0:
    return np.zeros((nworld,), dtype=np.float32)
  dist = d.contact.dist.numpy()[:nacon]
  wid = d.contact.worldid.numpy()[:nacon].astype(np.int64, copy=False)
  valid = (wid >= 0) & (wid < nworld) & np.isfinite(dist)
  if not np.any(valid):
    return np.zeros((nworld,), dtype=np.float32)
  per_world = np.full((nworld,), np.inf, dtype=np.float32)
  np.minimum.at(per_world, wid[valid], dist[valid].astype(np.float32))
  per_world[np.isinf(per_world)] = 0.0
  return per_world


def write_culprit_frame(mjm: mujoco.MjModel, m: mjw.Model, _mjd: mujoco.MjData, d: mjw.Data, world: int, out_path: str) -> None:
  renderer = mujoco.Renderer(mjm, width=640, height=480)
  d_vis = mjw.make_data(mjm, nworld=1)
  mjw.reset_data(m, d_vis)
  q0 = d.qpos.numpy()[world].reshape(1, -1).astype(np.float32)
  wp.copy(d_vis.qpos, wp.array(q0))
  mjw.forward(m, d_vis)
  mjw.get_data_into(_mjd, mjm, d_vis)
  renderer.update_scene(_mjd)
  frame = renderer.render()
  imageio.imwrite(out_path, frame)


def run(args: argparse.Namespace) -> int:
  rng = random.Random(args.seed)

  xml_rel = "benchmark/hand_box/scene.xml"
  mjm, _mjd = load_model(xml_rel)

  wp.init()
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=args.nworld, nconmax=64)
  mjw.reset_data(m, d)

  print(f"nworld={args.nworld} scene={xml_rel}")

  with wp.ScopedCapture() as cap:
    mjw.step(m, d)
  graph = cap.graph

  for ep in range(args.episodes):
    mjw.reset_data(m, d)

    for step in range(args.max_steps):
      wp.capture_launch(graph)
      wp.synchronize()

      w_nan = find_nan_world(d)
      if w_nan is not None:
        out = args.frame_out or "anomaly.png"
        write_culprit_frame(mjm, m, _mjd, d, w_nan, out)
        print(f"nan detected at ep={ep} step={step} world={w_nan} frame={out}")
        return 2

      per_world = min_penetration_by_world(d)
      if per_world.size and float(np.min(per_world)) < -args.max_penetration:
        w = int(np.argmin(per_world))
        out = args.frame_out or "anomaly.png"
        write_culprit_frame(mjm, m, _mjd, d, w, out)
        print(f"penetration detected at ep={ep} step={step} world={w} min={float(per_world[w]):.6f} frame={out}")
        return 2

    if (ep + 1) % 10 == 0:
      print(f"completed ep={ep+1}")

  print("no anomalies detected")
  return 0


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--episodes", type=int, default=100)
  p.add_argument("--max-steps", type=int, default=2000)
  p.add_argument("--nworld", type=int, default=1024)
  p.add_argument("--max-penetration", type=float, default=0.1)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--frame-out", type=str, default="", help="output image path for anomaly frame")
  return p.parse_args()


if __name__ == "__main__":
  exit(run(parse_args()))



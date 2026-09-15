# Copyright 2026 The Newton Developers
# Licensed under the Apache License, Version 2.0.
"""Compare Warp and OptiX batch rendering, including acceleration-structure updates."""

import argparse
import gc
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


def _measure(function, iterations, repeats):
  function()
  wp.synchronize()
  for _ in range(3):
    function()
  wp.synchronize()
  start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
  samples = []
  for _ in range(repeats):
    wp.record_event(start)
    for _ in range(iterations):
      function()
    wp.record_event(end)
    samples.append(wp.get_event_elapsed_time(start, end) / iterations)
  return {"median_ms": float(np.median(samples)), "min_ms": min(samples), "max_ms": max(samples)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("models", nargs="+", type=Path)
  parser.add_argument("--worlds", nargs="+", type=int, default=[1, 16, 128])
  parser.add_argument("--resolution", type=int, default=128)
  parser.add_argument("--shadows", action="store_true")
  parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--seg", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--cameras", nargs="+", help="Camera names; defaults to the first camera")
  parser.add_argument("--replay", type=Path, help="Use the initial qpos from a benchmark replay")
  parser.add_argument("--images", type=Path, help="Save first-world RGB comparisons for every active camera")
  parser.add_argument("--warp-graph", action="store_true", help="Also measure warmed Warp graph replay")
  parser.add_argument("--pose-noise", type=float, default=0.0, help="Independent hinge-angle noise per world, in radians")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--repeats", type=int, default=7)
  parser.add_argument("--output", type=Path, default=Path("render_backends.json"))
  args = parser.parse_args()
  if min(*args.worlds, args.resolution, args.iterations, args.repeats) < 1:
    parser.error("world counts, resolution, iterations, and repeats must be positive")
  if not np.isfinite(args.pose_noise) or args.pose_noise < 0:
    parser.error("pose noise must be finite and nonnegative")
  wp.init()
  results = {
    "gpu": wp.get_device().name,
    "warp": wp.__version__,
    "mujoco": mujoco.__version__,
    "resolution": [args.resolution] * 2,
    "shadows": args.shadows,
    "depth": args.depth,
    "segmentation": args.seg,
    "cameras": args.cameras,
    "replay": str(args.replay) if args.replay else None,
    "pose_noise_radians": args.pose_noise,
    "seed": args.seed,
    "mode": "eager CUDA-event latency including host submission",
    "iterations": args.iterations,
    "repeats": args.repeats,
    "results": [],
  }
  for path in args.models:
    mjm = mujoco.MjModel.from_xml_path(str(path.resolve()))
    if not mjm.ncam:
      raise ValueError(f"{path} has no camera")
    mjd = mujoco.MjData(mjm)
    if args.replay:
      with np.load(args.replay) as replay:
        mjd.qpos[:] = replay["qpos"][0]
    mujoco.mj_forward(mjm, mjd)
    cameras = [mjm.camera(name).id for name in args.cameras] if args.cameras else [0]
    for worlds in args.worlds:
      m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd, nworld=worlds)
      if args.pose_noise:
        qpos = d.qpos.numpy()
        rng = np.random.default_rng(args.seed)
        for joint in np.flatnonzero(mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE):
          adr = mjm.jnt_qposadr[joint]
          qpos[:, adr] += rng.normal(0.0, args.pose_noise, worlds)
          if mjm.jnt_limited[joint]:
            qpos[:, adr] = np.clip(qpos[:, adr], *mjm.jnt_range[joint])
        d.qpos.assign(qpos)
        mjw.fwd_kinematics(m, d)
      reference = None
      for backend in ("warp", "optix"):
        start = time.perf_counter()
        rc = mjw.create_render_context(
          mjm,
          nworld=worlds,
          backend=backend,
          cam_res=(args.resolution, args.resolution),
          cam_active=cameras,
          render_rgb=True,
          render_depth=args.depth,
          render_seg=args.seg,
          use_shadows=args.shadows,
        )
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        wp.synchronize()
        setup = time.perf_counter() - start
        image = [rc.rgb_data.numpy(), rc.depth_data.numpy(), rc.seg_data.numpy()]
        parity = {}
        if reference is None:
          reference = image
        else:
          # Compare unpacked channels; a one-level red difference is 65536 in packed RGB.
          rgb_error = max(
            int(np.max(np.abs(((image[0] >> shift) & 255).astype(int) - ((reference[0] >> shift) & 255).astype(int))))
            for shift in (0, 8, 16)
          )
          parity = {
            "rgb_max_channel_error": rgb_error,
            "depth_max_abs_error": float(np.max(np.abs(image[1] - reference[1]))) if args.depth else None,
            "segmentation_pixel_mismatches": (
              int(np.count_nonzero(np.any(image[2] != reference[2], axis=-1))) if args.seg else None
            ),
          }
          if args.images:
            from PIL import Image
            from PIL import ImageDraw

            args.images.mkdir(parents=True, exist_ok=True)
            size = args.resolution
            for index, camera in enumerate(cameras):
              panels = []
              for label, pixels in (("Warp", reference[0]), ("OptiX", image[0])):
                packed = pixels[0, index * size * size : (index + 1) * size * size].reshape(size, size)
                rgb = np.stack([(packed >> shift) & 255 for shift in (16, 8, 0)], axis=-1).astype(np.uint8)
                panel = Image.new("RGB", (size, size + 24), "#202020")
                panel.paste(Image.fromarray(rgb), (0, 24))
                ImageDraw.Draw(panel).text((6, 6), label, fill="white")
                panels.append(panel)
              comparison = Image.new("RGB", (size * 2, size + 24))
              for column, panel in enumerate(panels):
                comparison.paste(panel, (column * size, 0))
              camera_name = mjm.camera(camera).name or f"camera_{camera}"
              comparison.save(args.images / f"{path.stem}_{worlds}w_{camera_name}.png")

        def refit():
          mjw.refit_bvh(m, d, rc)

        def render():
          mjw.render(m, d, rc)

        def frame():
          refit()
          render()

        row = {
          "model": str(path),
          "worlds": worlds,
          "backend": backend,
          "ngeoms": mjm.ngeom,
          "mesh_triangles": mjm.nmeshface,
          "hardware_traversal": backend == "optix" and bool(rc._optix.geom_ids.size),
          "setup_seconds": setup,
          "parity": parity,
        }
        for name, fn in (("render", render), ("refit", refit), ("frame", frame)):
          row[name] = _measure(fn, args.iterations, args.repeats)
          if backend == "warp" and args.warp_graph:
            with wp.ScopedCapture() as capture:
              fn()
            row[name + "_graph"] = _measure(lambda: wp.capture_launch(capture.graph), args.iterations, args.repeats)
        results["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
          f"{path.name} worlds={worlds} {backend}: render={row['render']['median_ms']:.3f} ms "
          f"refit={row['refit']['median_ms']:.3f} ms frame={row['frame']['median_ms']:.3f} ms {parity}",
          flush=True,
        )
        del rc
        gc.collect()
      del m, d
      gc.collect()


if __name__ == "__main__":
  main()

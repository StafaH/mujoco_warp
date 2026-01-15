# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""mjwarp-render: render an RGB and/or depth image from an MJCF.

Usage: mjwarp-render <mjcf XML path> [flags]

Example:
  mjwarp-render benchmark/humanoid/humanoid.xml --nworld=1 --cam=0 --width=512 --height=512
"""

import sys
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath
from PIL import Image

import mujoco_warp as mjw
from mujoco_warp._src.io import override_model

_NWORLD = flags.DEFINE_integer("nworld", 2, "number of parallel worlds")
_WORLD = flags.DEFINE_integer("world", 0, "world index to save from")
_CAM = flags.DEFINE_integer("cam", 0, "camera index to render")
_WIDTH = flags.DEFINE_integer("width", 64, "render width (pixels)")
_HEIGHT = flags.DEFINE_integer("height", 64, "render height (pixels)")
_RENDER_RGB = flags.DEFINE_bool("rgb", True, "render RGB image")
_RENDER_DEPTH = flags.DEFINE_bool("depth", True, "render depth image")
_USE_TEXTURES = flags.DEFINE_bool("textures", True, "use textures")
_USE_SHADOWS = flags.DEFINE_bool("shadows", False, "use shadows")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "clear Warp kernel cache before rendering")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_OUTPUT_RGB = flags.DEFINE_string("output_rgb", "debug.png", "output path for RGB image")
_OUTPUT_DEPTH = flags.DEFINE_string("output_depth", "debug_depth.png", "output path for depth image")
_DEPTH_SCALE = flags.DEFINE_float("depth_scale", 5.0, "scale factor to map depth to 0..255 for preview")
_TILED = flags.DEFINE_bool("tiled", False, "render a 4x4 tiled grid across 16 worlds at 512x512")
_ROLLOUT = flags.DEFINE_bool("rollout", False, "render a rollout video instead of a single frame")
_NSTEPS = flags.DEFINE_integer("nstep", 128, "number of simulation steps in the rollout")
_ROLLOUT_OUTPUT = flags.DEFINE_string("output_video", "rollout.gif", "output path for rollout video")

def _load_model(path: epath.Path) -> mujoco.MjModel:
    if not path.exists():
        resource_path = epath.resource_path("mujoco_warp") / path
        if not resource_path.exists():
            raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
        path = resource_path

    print(f"Loading model from: {path}...")
    if path.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(path.as_posix())

    spec = mujoco.MjSpec.from_file(path.as_posix())
    # register SDF test plugins if present
    if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
        from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

        register_sdf_plugins(mjw)

    return spec.compile()


def _rgb_image_from_vec4(vec4_array: np.ndarray) -> np.ndarray:
  """Convert a (H, W, 4) vec4 array to an (H, W, 3) uint8 RGB array.

  The vec4 values are expected to be in [0.0, 1.0] range.
  """
  # vec4_array shape is (H, W, 4) where the 4 channels are RGBA in [0.0, 1.0]
  rgb = vec4_array[:, :, :3]  # Take RGB, drop alpha
  rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
  return rgb_uint8


def _save_rgb_from_vec4(vec4_array: np.ndarray, out_path: str):
  """Save a (H, W, 4) vec4 array as an RGB image."""
  rgb_uint8 = _rgb_image_from_vec4(vec4_array)
  img = Image.fromarray(rgb_uint8)
  img.save(out_path)


def _depth_image_from_array(depth_array: np.ndarray, scale: float) -> np.ndarray:
  """Convert a (H, W) depth array into an (H, W) uint8 array using the given scale."""
  arr = np.clip(depth_array / max(scale, 1e-6), 0.0, 1.0)
  return (arr * 255.0).astype(np.uint8)


def _save_depth(depth_array: np.ndarray, scale: float, out_path: str):
  """Save a (H, W) depth array as a grayscale image."""
  img = Image.fromarray(_depth_image_from_array(depth_array, scale))
  img.save(out_path)


def _save_tiled_rgb(
  rgb_4d: np.ndarray,
  cam_idx: int,
  grid_rows: int,
  grid_cols: int,
  out_path: str,
):
  """Tile multiple RGB worlds into a single image and save it.

  Args:
    rgb_4d: 4D array of shape (nworld, ncam, H, W, 4) with vec4 RGB values.
    cam_idx: Camera index to extract.
    grid_rows: Number of rows in the tile grid.
    grid_cols: Number of columns in the tile grid.
    out_path: Output file path.
  """
  nworld = rgb_4d.shape[0]
  expected = grid_rows * grid_cols
  if nworld < expected:
    raise ValueError(f"tiled rendering requires at least {expected} worlds, got {nworld}")

  tiles = []
  for wi in range(expected):
    # rgb_4d[wi, cam_idx] has shape (H, W, 4)
    tiles.append(_rgb_image_from_vec4(rgb_4d[wi, cam_idx]))

  rows = []
  for r in range(grid_rows):
    row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
    rows.append(np.concatenate(row_tiles, axis=1))
  full = np.concatenate(rows, axis=0)
  Image.fromarray(full).save(out_path)


def _save_tiled_depth(
  depth_4d: np.ndarray,
  cam_idx: int,
  scale: float,
  grid_rows: int,
  grid_cols: int,
  out_path: str,
):
  """Tile multiple depth worlds into a single image and save it.

  Args:
    depth_4d: 4D array of shape (nworld, ncam, H, W) with depth values.
    cam_idx: Camera index to extract.
    scale: Scale factor for depth visualization.
    grid_rows: Number of rows in the tile grid.
    grid_cols: Number of columns in the tile grid.
    out_path: Output file path.
  """
  nworld = depth_4d.shape[0]
  expected = grid_rows * grid_cols
  if nworld < expected:
    raise ValueError(f"tiled rendering requires at least {expected} worlds, got {nworld}")

  tiles = []
  for wi in range(expected):
    # depth_4d[wi, cam_idx] has shape (H, W)
    tiles.append(_depth_image_from_array(depth_4d[wi, cam_idx], scale))

  rows = []
  for r in range(grid_rows):
    row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
    rows.append(np.concatenate(row_tiles, axis=1))
  full = np.concatenate(rows, axis=0)
  Image.fromarray(full).save(out_path)


def _main(argv: Sequence[str]):
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  mjm = _load_model(epath.Path(argv[1]))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  with wp.ScopedDevice(_DEVICE.value):
    m = mjw.put_model(mjm)

    if _OVERRIDE.value:
      override_model(m, _OVERRIDE.value)

    # Configure parallel worlds and per-camera resolution.
    if _TILED.value:
      # In tiled mode we always use 16 worlds and output a 4x4 grid at 512x512.
      nworld = 16
      grid_rows = 4
      grid_cols = 4
      final_width = 512
      final_height = 512
      render_width = final_width // grid_cols
      render_height = final_height // grid_rows
    else:
      nworld = int(_NWORLD.value)
      grid_rows = grid_cols = 1
      render_width = int(_WIDTH.value)
      render_height = int(_HEIGHT.value)

    d = mjw.put_data(mjm, mjd, nworld=nworld)

    rc = mjw.create_render_context(
      mjm,
      m,
      d,
      (render_width, render_height),
      _RENDER_RGB.value,
      _RENDER_DEPTH.value,
      _USE_TEXTURES.value,
      _USE_SHADOWS.value,
      enabled_geom_groups=[0, 1, 2],
    )

    print(f"Model: ncam={m.ncam} nlight={m.nlight} ngeom={m.ngeom}\n")

    world = int(_WORLD.value)
    cam = int(_CAM.value)
    if cam < 0 or cam >= rc.ncam:
      raise ValueError(f"camera index out of range: {cam} not in [0, {rc.ncam - 1}]")
    if not _TILED.value:
      if world < 0 or world >= d.nworld:
        raise ValueError(f"world index out of range: {world} not in [0, {d.nworld - 1}]")

    # Output format: 4D arrays (nworld, ncam, H, W) with vec4 for RGB, float for depth

    if _ROLLOUT.value:
      if not _RENDER_RGB.value:
        raise ValueError("rollout video requires RGB rendering to be enabled (--rgb).")

      # Use the physics timestep to choose how many simulation steps each
      # video frame should cover so that playback is approximately realtime.
      try:
        dt = float(m.opt.timestep.numpy()[0])
      except Exception:
        dt = 1.0 / 60.0

      target_fps = 30.0
      steps_per_frame = max(1, int(round(1.0 / (dt * target_fps))))
      frame_duration_ms = max(1, int(round(1000.0 / target_fps)))

      total_steps = int(_NSTEPS.value)
      print(f"Rendering rollout for {total_steps} steps (dt={dt:.4f}, steps_per_frame={steps_per_frame})...")
      frames = []

      step = 0
      while step < total_steps:
        mjw.render(m, d, rc)

        # Get RGB data: shape (nworld, ncam, H, W, 4) for vec4
        rgb_all = rc.rgb_data.numpy()

        if _TILED.value:
          # Build a tiled frame from all worlds.
          expected = grid_rows * grid_cols
          if rgb_all.shape[0] >= expected:
            tiles = []
            for wi in range(expected):
              # rgb_all[wi, cam] has shape (H, W, 4)
              tiles.append(_rgb_image_from_vec4(rgb_all[wi, cam]))
            row_imgs = []
            for r in range(grid_rows):
              row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
              row_imgs.append(np.concatenate(row_tiles, axis=1))
            frame_array = np.concatenate(row_imgs, axis=0)
          else:
            frame_array = None
        else:
          # rgb_all[world, cam] has shape (H, W, 4)
          frame_array = _rgb_image_from_vec4(rgb_all[world, cam])

        if frame_array is not None:
          frames.append(Image.fromarray(frame_array))

        # Advance simulation by the number of steps represented by this frame.
        for _ in range(steps_per_frame):
          if step >= total_steps:
            break
          mjw.step(m, d)
          step += 1

      if not frames:
        raise RuntimeError("no RGB frames were generated during rollout")

      frames[0].save(
        _ROLLOUT_OUTPUT.value,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
      )
      print(f"Saved rollout video to: {_ROLLOUT_OUTPUT.value}")
      return

    # Single-frame rendering path.
    print("Rendering single frame...")
    mjw.render(m, d, rc)

    if _TILED.value:
      # Use all worlds and tile them into a 4x4 grid.
      if _RENDER_RGB.value:
        rgb_all = rc.rgb_data.numpy()
        _save_tiled_rgb(rgb_all, cam, grid_rows, grid_cols, _OUTPUT_RGB.value)
        print(f"Saved tiled RGB to: {_OUTPUT_RGB.value}")

      if _RENDER_DEPTH.value:
        depth_all = rc.depth_data.numpy()
        _save_tiled_depth(depth_all, cam, _DEPTH_SCALE.value, grid_rows, grid_cols, _OUTPUT_DEPTH.value)
        print(f"Saved tiled depth to: {_OUTPUT_DEPTH.value}")
    else:
      # Original single-world behavior.
      if _RENDER_RGB.value:
        rgb = rc.rgb_data.numpy()
        # rgb[world, cam] has shape (H, W, 4)
        _save_rgb_from_vec4(rgb[world, cam], _OUTPUT_RGB.value)
        print(f"Saved RGB to: {_OUTPUT_RGB.value}")

      if _RENDER_DEPTH.value:
        depth = rc.depth_data.numpy()
        # depth[world, cam] has shape (H, W)
        _save_depth(depth[world, cam], _DEPTH_SCALE.value, _OUTPUT_DEPTH.value)
        print(f"Saved depth to: {_OUTPUT_DEPTH.value}")


def main():
    sys.argv[0] = "mujoco_warp.render"
    sys.modules["__main__"].__doc__ = __doc__
    app.run(_main)


if __name__ == "__main__":
    main()

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

"""Render benchmarking script for MuJoCo Warp.

Measures total rendering FPS (refit_bvh + render) across scenes, world counts,
and resolutions, then produces a single figure with all scenes in a 2x3 grid
with camera preview images.

Usage:
  python renderbench.py
"""

import gc
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mujoco
import numpy as np
import seaborn as sns
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.benchmark import benchmark
from mujoco_warp._src.io import find_keys

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENES = [
    ("Cartpole", "benchmarks/cartpole_1cam.xml"),
    ("Cartpole 2", "benchmarks/cartpole_2cam.xml"),
    ("Primitives Scene", "benchmarks/primitives.xml"),
    ("Franka Panda Visual", "benchmarks/franka_emika_panda/scene.xml"),
    ("Franka Panda Primitive", "benchmarks/franka_primitive/scene.xml"),
    ("Franka Panda Primitive 2", "benchmarks/franka_primtive_2cam/scene.xml"),
    ("Apptronik Heightfield", "benchmarks/apptronik_apollo/scene_hfield.xml"),
]

RESOLUTIONS = [(32, 32), (64, 64), (128, 128), (256, 256)]
NWORLDS = [512, 1024, 2048, 4096]
NSTEPS = 100

NROWS, NCOLS = 4, 2

SCRIPT_DIR = Path(__file__).resolve().parent
RES_LABELS = [f"{r[0]}x{r[1]}" for r in RESOLUTIONS]

# ============================================================================
# Benchmarking
# ============================================================================


def _load_model(scene_path: str) -> mujoco.MjModel:
  path = os.path.join(SCRIPT_DIR, scene_path)
  spec = mujoco.MjSpec.from_file(path)
  return spec.compile()


def _reset_to_home(mjm: mujoco.MjModel, mjd: mujoco.MjData):
  keys = find_keys(mjm, "home")
  if keys:
    mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])
  mujoco.mj_forward(mjm, mjd)


def _render_preview(mjm: mujoco.MjModel) -> dict | None:
  """Render 256x256 RGB and depth previews from each camera.

  Returns dict with 'rgb' and 'depth' lists of arrays, or None if no cameras.
  """
  if mjm.ncam == 0:
    return None

  mjd = mujoco.MjData(mjm)
  _reset_to_home(mjm, mjd)

  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=1)
  rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(256, 256),
      render_rgb=True,
      render_depth=True,
      use_textures=True,
      use_shadows=False,
  )
  mjw.refit_bvh(m, d, rc)
  mjw.render(m, d, rc)
  wp.synchronize()

  rgb_all = rc.rgb_data.numpy()
  rgb_adr = rc.rgb_adr.numpy()
  depth_all = rc.depth_data.numpy()
  depth_adr = rc.depth_adr.numpy()
  cam_res = rc.cam_res.numpy()

  rgb_images = []
  depth_images = []
  for cam_idx in range(mjm.ncam):
    w, h = int(cam_res[cam_idx][0]), int(cam_res[cam_idx][1])
    npix = w * h

    # RGB
    adr = int(rgb_adr[cam_idx])
    packed = rgb_all[0, adr : adr + npix].reshape(h, w).astype(np.uint32)
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    b = (packed & 0xFF).astype(np.uint8)
    rgb_images.append(np.dstack([r, g, b]))

    # Depth (clamp to 95th percentile, normalize to 0-1)
    dadr = int(depth_adr[cam_idx])
    depth = depth_all[0, dadr : dadr + npix].reshape(h, w)
    valid = depth > 0
    if valid.any():
      dmax = np.percentile(depth[valid], 95)
      depth = np.clip(depth, 0, dmax)
      depth = np.where(valid, depth / dmax, 0.0)
    depth_images.append(depth)

  del m, d, rc
  gc.collect()
  wp.synchronize()

  return {"rgb": rgb_images, "depth": depth_images}


def _benchmark_scene(scene_path: str) -> dict:
  """Benchmark one scene, return {nworld: {res: total_fps}}."""
  mjm = _load_model(scene_path)
  print(
      f"  nbody={mjm.nbody}  ngeom={mjm.ngeom}  nv={mjm.nv}"
      f"  ncam={mjm.ncam}  nlight={mjm.nlight}"
  )

  def refit_and_render(m, d, rc):
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)

  results = {}
  with wp.ScopedDevice(device=None):
    for nw in NWORLDS:
      results[nw] = {}
      for res_w, res_h in RESOLUTIONS:
        res_key = f"{res_w}x{res_h}"
        print(f"  nworld={nw} {res_key} ...", end=" ", flush=True)

        mjd = mujoco.MjData(mjm)
        _reset_to_home(mjm, mjd)

        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nw)
        rc = mjw.create_render_context(
            mjm,
            nworld=nw,
            cam_res=(res_w, res_h),
            render_rgb=True,
            render_depth=True,
            use_textures=True,
            use_shadows=False,
        )

        _, total_time, *_ = benchmark(
            refit_and_render, m, d, NSTEPS, render_context=rc
        )

        fps = (nw * NSTEPS) / total_time
        results[nw][res_key] = fps
        print(f"{fps:,.1f} FPS")

        # Free GPU resources (textures, buffers) before next iteration.
        del m, d, rc
        gc.collect()
        wp.synchronize()

  return results


# ============================================================================
# Plotting
# ============================================================================


def plot_all(all_results: dict):
  """Plot all scenes in a single 2x3 grid figure with preview images."""
  sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
  palette = sns.color_palette("muted", len(RES_LABELS))

  n_scenes = len(all_results)
  total_slots = NROWS * NCOLS

  # Outer gridspec: 2x3 cells with spacing between cells.
  # Each cell gets a nested subgridspec [graph | preview] with tight spacing.
  fig = plt.figure(figsize=(16, 24))
  gs_outer = fig.add_gridspec(NROWS, NCOLS, wspace=0.35, hspace=0.45)

  n_res = len(RESOLUTIONS)
  x = np.arange(len(NWORLDS))
  bar_width = 0.65 / n_res

  # Scenes fill sequentially; legend goes in the last empty slot.
  scene_items = list(all_results.items())

  def _add_scene(row, col, label, scene_data):
    results = scene_data["fps"]
    previews = scene_data.get("preview_imgs")

    # Nested gridspec: graph + preview column (RGB then depth, stacked)
    if previews:
      n_cams = len(previews["rgb"])
      gs_cell = gs_outer[row, col].subgridspec(
          1, 2, width_ratios=[5, 1], wspace=0.05,
      )
      ax = fig.add_subplot(gs_cell[0])
      # Stack: RGB0, Depth0, RGB1, Depth1, ...
      gs_imgs = gs_cell[1].subgridspec(n_cams * 2, 1, hspace=0.6)
      for i in range(n_cams):
        ax_rgb = fig.add_subplot(gs_imgs[i * 2])
        ax_rgb.imshow(previews["rgb"][i])
        ax_rgb.set_axis_off()
        ax_rgb.set_title(f"Cam {i} RGB", fontsize=8, color="0.4")
        ax_depth = fig.add_subplot(gs_imgs[i * 2 + 1])
        ax_depth.imshow(previews["depth"][i], cmap="gray")
        ax_depth.set_axis_off()
        ax_depth.set_title(f"Cam {i} Depth", fontsize=8, color="0.4")
    else:
      ax = fig.add_subplot(gs_outer[row, col])

    for i, res in enumerate(RESOLUTIONS):
      res_key = f"{res[0]}x{res[1]}"
      fps = [results[nw][res_key] for nw in NWORLDS]
      offset = (i - n_res / 2 + 0.5) * bar_width
      bars = ax.bar(
          x + offset, fps, bar_width,
          label=RES_LABELS[i], color=palette[i],
          edgecolor="white", linewidth=0.6,
      )
      for bar, val in zip(bars, fps):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:,.0f}", ha="center", va="bottom", fontsize=7, color="0.3",
        )

    scene_y_max = max(
        fps_val
        for nw_data in results.values()
        for fps_val in nw_data.values()
    ) * 1.15
    ax.set_ylim(0, scene_y_max)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nw:,}" for nw in NWORLDS])
    ax.set_xlabel("Number of worlds")
    sns.despine(ax=ax, left=True)
    if col == 0:
      ax.set_ylabel("FPS")
    return ax

  graph_axes = []
  for scene_idx, (label, scene_data) in enumerate(scene_items):
    row = scene_idx // NCOLS
    col = scene_idx % NCOLS
    ax = _add_scene(row, col, label, scene_data)
    graph_axes.append(ax)

  # Legend in the bottom-right slot (last row, last column)
  handles, labels = graph_axes[0].get_legend_handles_labels()
  ax_legend = fig.add_subplot(gs_outer[NROWS - 1, NCOLS - 1])
  ax_legend.set_axis_off()
  ax_legend.legend(
      handles, labels,
      title="Resolution", loc="center",
      frameon=True, fancybox=True, framealpha=0.9,
      fontsize=14, title_fontsize=15,
  )

  fname = "render_benchmark.png"
  fig.savefig(fname, dpi=150, bbox_inches="tight")
  print(f"\nSaved {fname}")
  plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
  matplotlib.use("Agg")
  wp.config.quiet = True
  wp.init()

  # Force warp's memory pool to release freed memory back to CUDA immediately,
  # preventing OOM from accumulated texture/buffer allocations across scenes.
  wp.set_mempool_release_threshold("cuda:0", 0)

  all_results = {}
  for label, scene_path in SCENES:
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label}")
    print(f"{'=' * 60}")

    mjm = _load_model(scene_path)
    fps = _benchmark_scene(scene_path)
    previews = _render_preview(mjm)

    # Build title with camera/light info
    title = f"{label} ({mjm.ncam} cam, {mjm.nlight} light"
    if mjm.ncam != 1:
      title = f"{label} ({mjm.ncam} cams, "
    else:
      title = f"{label} ({mjm.ncam} cam, "
    if mjm.nlight != 1:
      title += f"{mjm.nlight} lights)"
    else:
      title += f"{mjm.nlight} light)"

    # Save preview images
    if previews:
      for i, img in enumerate(previews["rgb"]):
        fname = f"render_preview_{label}_cam{i}_rgb.png"
        plt.imsave(fname, img)
        print(f"Saved {fname}")
      for i, img in enumerate(previews["depth"]):
        fname = f"render_preview_{label}_cam{i}_depth.png"
        plt.imsave(fname, img, cmap="gray")
        print(f"Saved {fname}")

    all_results[title] = {"fps": fps, "preview_imgs": previews}

  plot_all(all_results)
  print("Done.")


if __name__ == "__main__":
  main()

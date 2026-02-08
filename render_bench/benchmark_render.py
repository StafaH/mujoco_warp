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

Measures rendering performance (refit_bvh + render) across different scenes,
world counts, and resolutions, then produces two plots per scene:

  1. Total FPS (refit + render combined).
  2. Stacked bar showing refit vs render time breakdown within total FPS.

Each scene is benchmarked in a separate subprocess to guarantee a clean GPU
context (CUDA graphs, BVH structures, kernel caches) between scenes.

Usage:
  python benchmark_render.py
"""

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.image import imread
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION - Edit these lists to change what gets benchmarked.
# ============================================================================

# Scenes to benchmark: (display_name, xml_path).
SCENES = [
    ("Cartpole 1-Cam", "benchmarks/cartpole_1cam.xml"),
    ("Cartpole 2-Cam", "benchmarks/cartpole_2cam.xml"),
    ("Primitives Scene", "benchmarks/primitives.xml"),
    ("Franka Panda Visual", "benchmarks/franka_emika_panda/scene.xml"),
    ("Franka Panda Primitive", "benchmarks/franka_primitive/scene.xml"),
    ("Franka Panda 2-Cam Primitive", "benchmarks/franka_primtive_2cam/scene.xml"),
    ("Apptronik Heightfield", "benchmarks/apptronik_apollo/scene_hfield.xml"),
]

# Resolutions to test (width, height).
RESOLUTIONS = [(32, 32), (64, 64), (128, 128), (256, 256)]

# Number of parallel worlds to sweep.
NWORLDS = [512, 1024, 2048, 4096, 8192]

# Number of steps per benchmark run.
NSTEPS = 100

# ============================================================================


def _create_figure_with_preview(preview_imgs):
  """Create a figure, optionally with preview image panels on the right.

  Args:
    preview_imgs: None, a single image array, or a list of image arrays.
  """
  if preview_imgs is None:
    fig, ax = plt.subplots(figsize=(10, 5))
    return fig, ax

  # Normalize to list
  if not isinstance(preview_imgs, list):
    preview_imgs = [preview_imgs]

  n_imgs = len(preview_imgs)
  if n_imgs == 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    return fig, ax

  fig = plt.figure(figsize=(12, 5))
  gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.08)
  ax = fig.add_subplot(gs[0])

  # Create nested gridspec for preview images (stacked vertically)
  gs_imgs = gs[1].subgridspec(n_imgs, 1, hspace=0.1)
  for i, img in enumerate(preview_imgs):
    ax_img = fig.add_subplot(gs_imgs[i])
    ax_img.imshow(img)
    ax_img.set_axis_off()
    ax_img.set_title(f"Cam {i}", fontsize=9, color="0.4")

  return fig, ax


def plot_total_fps(results, label, nworlds, resolutions, res_labels,
                   preview_img=None):
  """Plot 1: grouped bar chart of total FPS, grouped by nworld."""
  sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
  palette = sns.color_palette("muted", len(res_labels))

  n_worlds = len(nworlds)
  n_res = len(res_labels)
  x = np.arange(n_worlds)
  width = 0.65 / n_res

  fig, ax = _create_figure_with_preview(preview_img)

  for i, (res, rlabel) in enumerate(zip(resolutions, res_labels)):
    res_key = f"{res[0]}x{res[1]}"
    fps = [results[str(nw)][res_key]["total_fps"] for nw in nworlds]
    offset = (i - n_res / 2 + 0.5) * width
    bars = ax.bar(
        x + offset,
        fps,
        width,
        label=rlabel,
        color=palette[i],
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, val in zip(bars, fps):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height(),
          f"{val:,.0f}",
          ha="center",
          va="bottom",
          fontsize=7,
          color="0.3",
      )

  ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
  ax.set_xlabel("Number of worlds")
  ax.set_ylabel("FPS")
  ax.set_title(f"{label} — Total Rendering FPS (refit + render)")
  ax.set_xticks(x)
  ax.set_xticklabels([f"{nw:,}" for nw in nworlds])
  ax.legend(
      title="Resolution",
      loc="upper center",
      bbox_to_anchor=(0.5, -0.13),
      ncol=n_res,
      frameon=True, fancybox=True, framealpha=0.9,
  )
  sns.despine(left=True)
  fig.tight_layout()

  fname = f"render_benchmark_total_fps_{label}.png"
  fig.savefig(fname, dpi=150, bbox_inches="tight")
  print(f"Saved {fname}")
  plt.close(fig)


def plot_breakdown(results, label, nworlds, resolutions, res_labels,
                   preview_img=None):
  """Plot 2: stacked bar chart showing refit vs render proportions of FPS."""
  sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
  palette = sns.color_palette("muted", len(res_labels))

  n_worlds = len(nworlds)
  n_res = len(res_labels)
  x = np.arange(n_worlds)
  width = 0.65 / n_res

  fig, ax = _create_figure_with_preview(preview_img)

  for i, (res, rlabel) in enumerate(zip(resolutions, res_labels)):
    res_key = f"{res[0]}x{res[1]}"
    offset = (i - n_res / 2 + 0.5) * width

    total_fps = [results[str(nw)][res_key]["total_fps"] for nw in nworlds]
    refit_t = [results[str(nw)][res_key]["refit_time_per_frame"] for nw in nworlds]
    render_t = [results[str(nw)][res_key]["render_time_per_frame"] for nw in nworlds]

    refit_portions = [
        fps * (rt / (rt + rdt))
        for fps, rt, rdt in zip(total_fps, refit_t, render_t)
    ]
    render_portions = [
        fps * (rdt / (rt + rdt))
        for fps, rt, rdt in zip(total_fps, refit_t, render_t)
    ]

    ax.bar(
        x + offset,
        refit_portions,
        width,
        color=palette[i],
        alpha=0.40,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + offset,
        render_portions,
        width,
        bottom=refit_portions,
        color=palette[i],
        alpha=1.0,
        edgecolor="white",
        linewidth=0.6,
    )

  legend_elements = [
      Patch(facecolor="0.45", alpha=0.40, label="Refit BVH"),
      Patch(facecolor="0.45", alpha=1.0, label="Render"),
  ]
  for i, rlabel in enumerate(res_labels):
    legend_elements.append(Patch(facecolor=palette[i], label=rlabel))

  ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
  ax.set_xlabel("Number of worlds")
  ax.set_ylabel("FPS")
  ax.set_title(f"{label} — FPS Breakdown (Refit BVH + Render)")
  ax.set_xticks(x)
  ax.set_xticklabels([f"{nw:,}" for nw in nworlds])
  ax.legend(
      handles=legend_elements,
      title="Component / Resolution",
      loc="upper center",
      bbox_to_anchor=(0.5, -0.13),
      ncol=len(legend_elements),
      frameon=True,
      fancybox=True,
      framealpha=0.9,
  )
  sns.despine(left=True)
  fig.tight_layout()

  fname = f"render_benchmark_breakdown_{label}.png"
  fig.savefig(fname, dpi=150, bbox_inches="tight")
  print(f"Saved {fname}")
  plt.close(fig)


# ---- Worker: runs in a subprocess for a single scene ----

def _worker_main(scene_path, resolutions, nworlds, nsteps):
  """Benchmark one scene and print results as JSON to stdout."""
  import mujoco
  import warp as wp

  import mujoco_warp as mjw
  from mujoco_warp._src.benchmark import benchmark
  from mujoco_warp._src.io import find_keys

  def refit_and_render(m, d, rc):
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)

  # All XML paths are resolved relative to the script's directory
  # (render_bench/), so "benchmarks/foo.xml" -> "render_bench/benchmarks/foo.xml".
  script_dir = os.path.dirname(os.path.abspath(__file__))

  def load_model(path_str):
    path = os.path.join(script_dir, path_str)
    if not os.path.exists(path):
      raise FileNotFoundError(f"File not found: {path}")
    spec = mujoco.MjSpec.from_file(path)
    return spec.compile()

  wp.config.quiet = True
  wp.init()

  mjm = load_model(scene_path)
  print(
      f"  nbody={mjm.nbody}  ngeom={mjm.ngeom}  nv={mjm.nv}"
      f"  ncam={mjm.ncam}  nlight={mjm.nlight}",
      file=sys.stderr,
  )

  scene_results = {}
  with wp.ScopedDevice(device=None):
    for nw in nworlds:
      scene_results[str(nw)] = {}
      for res_w, res_h in resolutions:
        res_key = f"{res_w}x{res_h}"
        print(
            f"  nworld={nw} {res_key} ...",
            end=" ",
            flush=True,
            file=sys.stderr,
        )

        mjd = mujoco.MjData(mjm)
        keys = find_keys(mjm, "home")
        if keys:
          print(f"Found {len(keys)} keyframes", file=sys.stderr)
          print(f"Resetting to keyframe {keys[0]}", file=sys.stderr)
          mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])
          mujoco.mj_forward(mjm, mjd)

        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nw)
        rc = mjw.create_render_context(
            mjm,
            m,
            d,
            cam_res=(res_w, res_h),
            render_rgb=True,
            render_depth=True,
            use_textures=True,
            use_shadows=False,
        )

        _, refit_time, *_ = benchmark(
            mjw.refit_bvh, m, d, nsteps, render_context=rc
        )
        _, render_time, *_ = benchmark(
            mjw.render, m, d, nsteps, render_context=rc
        )
        _, total_time, *_ = benchmark(
            refit_and_render, m, d, nsteps, render_context=rc
        )

        steps = nw * nsteps
        metrics = {
            "refit_fps": steps / refit_time,
            "render_fps": steps / render_time,
            "total_fps": steps / total_time,
            "refit_time_per_frame": refit_time / steps,
            "render_time_per_frame": render_time / steps,
            "total_time_per_frame": total_time / steps,
        }
        scene_results[str(nw)][res_key] = metrics
        print(
            f"total {metrics['total_fps']:,.1f} FPS  "
            f"(refit {metrics['refit_fps']:,.1f}  "
            f"render {metrics['render_fps']:,.1f})",
            file=sys.stderr,
        )

    # Render a 256x256 preview image from each camera, world 0.
    preview_images_b64 = []
    if mjm.ncam > 0:
      import base64 as b64
      import io as _io

      print(f"  Rendering preview images for {mjm.ncam} camera(s) ...", file=sys.stderr)
      preview_res = (256, 256)
      mjd_prev = mujoco.MjData(mjm)
      mujoco.mj_forward(mjm, mjd_prev)
      keys = find_keys(mjm, "home")
      if keys:
        mujoco.mj_resetDataKeyframe(mjm, mjd_prev, keys[0])
        mujoco.mj_forward(mjm, mjd_prev)
      m_prev = mjw.put_model(mjm)
      d_prev = mjw.put_data(mjm, mjd_prev, nworld=1)
      rc_prev = mjw.create_render_context(
          mjm, m_prev, d_prev,
          cam_res=preview_res,
          render_rgb=True,
          render_depth=[False] * mjm.ncam,
          use_textures=True,
          use_shadows=False,
      )
      mjw.refit_bvh(m_prev, d_prev, rc_prev)
      mjw.render(m_prev, d_prev, rc_prev)
      wp.synchronize()

      rgb_all = rc_prev.rgb_data.numpy()
      rgb_adr = rc_prev.rgb_adr.numpy()
      cam_res_np = rc_prev.cam_res.numpy()

      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as _plt

      # Extract preview image for each camera
      for cam_idx in range(mjm.ncam):
        w, h = int(cam_res_np[cam_idx][0]), int(cam_res_np[cam_idx][1])
        adr = int(rgb_adr[cam_idx])
        packed = rgb_all[0, adr : adr + w * h].reshape(h, w).astype(np.uint32)
        r = ((packed >> 16) & 0xFF).astype(np.uint8)
        g = ((packed >> 8) & 0xFF).astype(np.uint8)
        b = (packed & 0xFF).astype(np.uint8)
        rgb_img = np.dstack([r, g, b])

        buf = _io.BytesIO()
        _plt.imsave(buf, rgb_img, format="png")
        preview_images_b64.append(b64.b64encode(buf.getvalue()).decode("ascii"))

      del m_prev, d_prev, rc_prev, mjd_prev

  # Emit results as a single JSON line on stdout.
  output = {"metrics": scene_results}
  if preview_images_b64:
    output["preview_images"] = preview_images_b64
  print(json.dumps(output))


# ---- Main: spawns one subprocess per scene ----

def _run_benchmark(label, scene_path, project_root, resolutions, nworlds, nsteps):
  """Run benchmark for a single scene and return results."""
  print(f"\n{'=' * 60}")
  print(f"Benchmarking: {label}")
  print(f"{'=' * 60}")

  resolutions_arg = json.dumps(resolutions)
  nworlds_arg = json.dumps(nworlds)

  worker_code = (
      f"import sys; sys.path.insert(0, {str(project_root)!r}); "
      f"from render_bench.benchmark_render import _worker_main; "
      f"_worker_main("
      f"  {scene_path!r},"
      f"  {resolutions_arg},"
      f"  {nworlds_arg},"
      f"  {nsteps},"
      f")"
  )

  result = subprocess.run(
      [sys.executable, "-c", worker_code],
      capture_output=True,
      text=True,
      cwd=str(project_root),
  )

  if result.stderr:
    sys.stderr.write(result.stderr)

  if result.returncode != 0:
    print(f"ERROR: benchmark for '{label}' failed (exit {result.returncode})")
    if result.stderr:
      print(result.stderr)
    return None

  output = json.loads(result.stdout.strip())
  scene_results = output["metrics"]

  # Decode preview images if present.
  preview_imgs = None
  if "preview_images" in output:
    preview_imgs = []
    for i, img_b64 in enumerate(output["preview_images"]):
      png_bytes = base64.b64decode(img_b64)
      preview_imgs.append(imread(io.BytesIO(png_bytes), format="png"))
      img_fname = f"render_preview_{label}_cam{i}.png"
      with open(img_fname, "wb") as f:
        f.write(png_bytes)
      print(f"Saved {img_fname}")

  return {"metrics": scene_results, "preview_imgs": preview_imgs}


def _get_scene_group(label):
  """Get the group name for a scene based on its first word."""
  return label.split()[0]


def plot_grouped_total_fps(all_results, group_name, nworlds, resolutions, res_labels):
  """Plot grouped bar chart of total FPS for multiple scenes with shared y-axis."""
  sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

  n_scenes = len(all_results)
  n_worlds = len(nworlds)
  n_res = len(res_labels)

  # Count scenes with preview images
  n_with_previews = sum(1 for data in all_results.values() if data["preview_imgs"])

  # Create figure - each scene group (graph + optional preview) gets space
  fig = plt.figure(figsize=(7 * n_scenes + 1.5 * n_with_previews, 5))

  # Outer gridspec: one column per scene group, with spacing between groups
  gs_outer = fig.add_gridspec(1, n_scenes, wspace=0.4)

  axes = []
  for scene_idx, (label, data) in enumerate(all_results.items()):
    has_previews = data["preview_imgs"] and len(data["preview_imgs"]) > 0

    if has_previews:
      # Inner gridspec: graph (wider) + preview column (narrower), tight spacing
      gs_inner = gs_outer[scene_idx].subgridspec(1, 2, width_ratios=[6, 1.5], wspace=0.08)
      ax = fig.add_subplot(gs_inner[0])
      axes.append(ax)

      # Add preview images stacked vertically
      gs_imgs = gs_inner[1].subgridspec(len(data["preview_imgs"]), 1, hspace=0.1)
      for i, img in enumerate(data["preview_imgs"]):
        ax_img = fig.add_subplot(gs_imgs[i])
        ax_img.imshow(img)
        ax_img.set_axis_off()
        ax_img.set_title(f"Cam {i}", fontsize=9, color="0.4")
    else:
      # No previews - just the graph
      ax = fig.add_subplot(gs_outer[scene_idx])
      axes.append(ax)

  # Find global y-max for shared axis
  y_max = 0
  for label, data in all_results.items():
    results = data["metrics"]
    for res in resolutions:
      res_key = f"{res[0]}x{res[1]}"
      for nw in nworlds:
        y_max = max(y_max, results[str(nw)][res_key]["total_fps"])
  y_max *= 1.15  # Add headroom for labels

  palette = sns.color_palette("muted", n_res)
  x = np.arange(n_worlds)
  width = 0.65 / n_res

  for ax, (label, data) in zip(axes, all_results.items()):
    results = data["metrics"]

    for i, (res, rlabel) in enumerate(zip(resolutions, res_labels)):
      res_key = f"{res[0]}x{res[1]}"
      fps = [results[str(nw)][res_key]["total_fps"] for nw in nworlds]
      offset = (i - n_res / 2 + 0.5) * width
      bars = ax.bar(
          x + offset,
          fps,
          width,
          label=rlabel,
          color=palette[i],
          edgecolor="white",
          linewidth=0.6,
      )
      for bar, val in zip(bars, fps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.3",
        )

    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Number of worlds")
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nw:,}" for nw in nworlds])
    sns.despine(ax=ax, left=True)

  # Only show y-label on first subplot
  axes[0].set_ylabel("FPS")

  # Add shared legend
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles, labels,
      title="Resolution",
      loc="upper center",
      bbox_to_anchor=(0.5, -0.02),
      ncol=n_res,
      frameon=True, fancybox=True, framealpha=0.9,
  )

  fig.suptitle(f"{group_name} — Total Rendering FPS (refit + render)", fontsize=14)
  fig.tight_layout(rect=[0, 0.05, 1, 0.95])

  fname = f"render_benchmark_total_fps_{group_name}.png"
  fig.savefig(fname, dpi=150, bbox_inches="tight")
  print(f"Saved {fname}")
  plt.close(fig)


def plot_grouped_breakdown(all_results, group_name, nworlds, resolutions, res_labels):
  """Plot grouped stacked bar chart for multiple scenes with shared y-axis."""
  sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

  n_scenes = len(all_results)
  n_worlds = len(nworlds)
  n_res = len(res_labels)

  # Count scenes with preview images
  n_with_previews = sum(1 for data in all_results.values() if data["preview_imgs"])

  # Create figure - each scene group (graph + optional preview) gets space
  fig = plt.figure(figsize=(7 * n_scenes + 1.5 * n_with_previews, 5))

  # Outer gridspec: one column per scene group, with spacing between groups
  gs_outer = fig.add_gridspec(1, n_scenes, wspace=0.4)

  axes = []
  for scene_idx, (label, data) in enumerate(all_results.items()):
    has_previews = data["preview_imgs"] and len(data["preview_imgs"]) > 0

    if has_previews:
      # Inner gridspec: graph (wider) + preview column (narrower), tight spacing
      gs_inner = gs_outer[scene_idx].subgridspec(1, 2, width_ratios=[6, 1.5], wspace=0.08)
      ax = fig.add_subplot(gs_inner[0])
      axes.append(ax)

      # Add preview images stacked vertically
      gs_imgs = gs_inner[1].subgridspec(len(data["preview_imgs"]), 1, hspace=0.1)
      for i, img in enumerate(data["preview_imgs"]):
        ax_img = fig.add_subplot(gs_imgs[i])
        ax_img.imshow(img)
        ax_img.set_axis_off()
        ax_img.set_title(f"Cam {i}", fontsize=9, color="0.4")
    else:
      # No previews - just the graph
      ax = fig.add_subplot(gs_outer[scene_idx])
      axes.append(ax)

  # Find global y-max for shared axis
  y_max = 0
  for label, data in all_results.items():
    results = data["metrics"]
    for res in resolutions:
      res_key = f"{res[0]}x{res[1]}"
      for nw in nworlds:
        y_max = max(y_max, results[str(nw)][res_key]["total_fps"])
  y_max *= 1.15

  palette = sns.color_palette("muted", n_res)
  x = np.arange(n_worlds)
  width = 0.65 / n_res

  for ax, (label, data) in zip(axes, all_results.items()):
    results = data["metrics"]

    for i, (res, rlabel) in enumerate(zip(resolutions, res_labels)):
      res_key = f"{res[0]}x{res[1]}"
      offset = (i - n_res / 2 + 0.5) * width

      total_fps = [results[str(nw)][res_key]["total_fps"] for nw in nworlds]
      refit_t = [results[str(nw)][res_key]["refit_time_per_frame"] for nw in nworlds]
      render_t = [results[str(nw)][res_key]["render_time_per_frame"] for nw in nworlds]

      refit_portions = [
          fps * (rt / (rt + rdt))
          for fps, rt, rdt in zip(total_fps, refit_t, render_t)
      ]
      render_portions = [
          fps * (rdt / (rt + rdt))
          for fps, rt, rdt in zip(total_fps, refit_t, render_t)
      ]

      ax.bar(
          x + offset,
          refit_portions,
          width,
          color=palette[i],
          alpha=0.40,
          edgecolor="white",
          linewidth=0.6,
      )
      ax.bar(
          x + offset,
          render_portions,
          width,
          bottom=refit_portions,
          color=palette[i],
          alpha=1.0,
          edgecolor="white",
          linewidth=0.6,
      )

    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Number of worlds")
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nw:,}" for nw in nworlds])
    sns.despine(ax=ax, left=True)

  axes[0].set_ylabel("FPS")

  # Build legend
  legend_elements = [
      Patch(facecolor="0.45", alpha=0.40, label="Refit BVH"),
      Patch(facecolor="0.45", alpha=1.0, label="Render"),
  ]
  for i, rlabel in enumerate(res_labels):
    legend_elements.append(Patch(facecolor=palette[i], label=rlabel))

  fig.legend(
      handles=legend_elements,
      title="Component / Resolution",
      loc="upper center",
      bbox_to_anchor=(0.5, -0.02),
      ncol=len(legend_elements),
      frameon=True,
      fancybox=True,
      framealpha=0.9,
  )

  fig.suptitle(f"{group_name} — FPS Breakdown (Refit BVH + Render)", fontsize=14)
  fig.tight_layout(rect=[0, 0.05, 1, 0.95])

  fname = f"render_benchmark_breakdown_{group_name}.png"
  fig.savefig(fname, dpi=150, bbox_inches="tight")
  print(f"Saved {fname}")
  plt.close(fig)


def main():
  # Resolve project root (parent of render_bench/) so subprocesses can find
  # both render_bench and mujoco_warp regardless of the caller's cwd.
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  # Run all benchmarks and collect results
  all_scene_data = {}
  for label, scene_path in SCENES:
    result = _run_benchmark(label, scene_path, project_root, RESOLUTIONS, NWORLDS, NSTEPS)
    if result is not None:
      all_scene_data[label] = result

  # Group scenes by first word
  from collections import OrderedDict
  groups = OrderedDict()
  for label in all_scene_data:
    group = _get_scene_group(label)
    if group not in groups:
      groups[group] = OrderedDict()
    groups[group][label] = all_scene_data[label]

  res_labels = [f"{r[0]}x{r[1]}" for r in RESOLUTIONS]

  # Plot each group
  for group_name, group_data in groups.items():
    if len(group_data) == 1:
      # Single scene in group - use original plotting functions
      label = list(group_data.keys())[0]
      data = group_data[label]
      plot_total_fps(data["metrics"], label, NWORLDS, RESOLUTIONS, res_labels,
                     preview_img=data["preview_imgs"])
      plot_breakdown(data["metrics"], label, NWORLDS, RESOLUTIONS, res_labels,
                     preview_img=data["preview_imgs"])
    else:
      # Multiple scenes in group - use grouped plotting functions
      plot_grouped_total_fps(group_data, group_name, NWORLDS, RESOLUTIONS, res_labels)
      plot_grouped_breakdown(group_data, group_name, NWORLDS, RESOLUTIONS, res_labels)

  print("\nDone.")


if __name__ == "__main__":
  main()

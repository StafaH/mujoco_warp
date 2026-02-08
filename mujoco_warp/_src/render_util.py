# Copyright 2026 The Newton Developers
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

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src.types import ProjectionType

wp.set_module_options({"enable_backward": False})


def _extract_texture_rgba(mjm: mujoco.MjModel, tex_id: int) -> np.ndarray:
  """Extract MuJoCo texture data as an (H, W, 4) uint8 RGBA numpy array."""
  tex_adr = mjm.tex_adr[tex_id]
  w = mjm.tex_width[tex_id]
  h = mjm.tex_height[tex_id]
  nc = mjm.tex_nchannel[tex_id]

  raw = mjm.tex_data[tex_adr : tex_adr + w * h * nc].reshape(h, w, nc)
  rgba = np.full((h, w, 4), 255, dtype=np.uint8)
  rgba[:, :, :nc] = raw
  return rgba


def _downsample_rgba(data: np.ndarray) -> np.ndarray:
  """Downsample an (H, W, 4) uint8 array by 2x using box filter."""
  h, w = data.shape[:2]
  new_h = max(h // 2, 1)
  new_w = max(w // 2, 1)

  # Average in float to avoid uint8 overflow, handling dimensions that are
  # already 1 (only halve in the other direction).
  fdata = data.astype(np.float32)
  if h >= 2 and w >= 2:
    fdata = fdata[: new_h * 2, : new_w * 2]
    fdata = fdata.reshape(new_h, 2, new_w, 2, 4).mean(axis=(1, 3))
  elif h >= 2:  # w == 1
    fdata = fdata[: new_h * 2, :1]
    fdata = fdata.reshape(new_h, 2, 1, 4).mean(axis=1)
  elif w >= 2:  # h == 1
    fdata = fdata[:1, : new_w * 2]
    fdata = fdata.reshape(1, new_w, 2, 4).mean(axis=2)
  else:
    fdata = fdata[:1, :1]

  return (fdata + 0.5).astype(np.uint8)


def create_warp_texture(mjm: mujoco.MjModel, tex_id: int) -> wp.Texture2D:
  """Create a Warp Texture2D from MuJoCo texture data.

  Uses the hardware CUDA texture API with uint8 data (auto-normalized to
  [0, 1] floats), WRAP address mode for proper tiling, and LINEAR filtering.
  """
  rgba = _extract_texture_rgba(mjm, tex_id)
  return wp.Texture2D(
    rgba,
    filter_mode=wp.Texture.FILTER_LINEAR,
    address_mode=wp.Texture.ADDRESS_WRAP,
  )


def create_warp_mipmap_chain(
  mjm: mujoco.MjModel, tex_id: int,
) -> list[wp.Texture2D]:
  """Create a chain of Texture2D mipmap levels from a MuJoCo texture.

  Returns a list of Texture2D objects from level 0 (full resolution) down to
  1x1. Each level is created with WRAP address mode and LINEAR filtering so
  the hardware handles bilinear interpolation; the caller selects the
  appropriate mip level and interpolates between adjacent levels (trilinear).
  """
  data = _extract_texture_rgba(mjm, tex_id)

  mip_chain = [
    wp.Texture2D(
      data,
      filter_mode=wp.Texture.FILTER_LINEAR,
      address_mode=wp.Texture.ADDRESS_WRAP,
    )
  ]

  while data.shape[0] > 1 or data.shape[1] > 1:
    data = _downsample_rgba(data)
    mip_chain.append(
      wp.Texture2D(
        data,
        filter_mode=wp.Texture.FILTER_LINEAR,
        address_mode=wp.Texture.ADDRESS_WRAP,
      )
    )

  return mip_chain


@wp.func
def compute_ray(
  # In:
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  px: int,
  py: int,
  znear: float,
) -> wp.vec3:
  """Compute ray direction for a pixel with per-world camera parameters.

  This combines _camera_frustum_bounds and build_primary_rays logic for use
  inside a kernel when camera parameters are batched/randomized across worlds.
  """
  if projection == ProjectionType.ORTHOGRAPHIC:
    return wp.vec3(0.0, 0.0, -1.0)

  aspect = float(img_w) / float(img_h)
  sensor_h = sensorsize[1]

  # Check if we have intrinsics (sensorsize[1] != 0)
  if sensor_h != 0.0:
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    sensor_w = sensorsize[0]

    target_aspect = float(img_w) / float(img_h)
    sensor_aspect = sensor_w / sensor_h
    if target_aspect > sensor_aspect:
      sensor_h = sensor_w / target_aspect
    elif target_aspect < sensor_aspect:
      sensor_w = sensor_h * target_aspect

    inv_fx_znear = znear / fx
    inv_fy_znear = znear / fy
    left = -inv_fx_znear * (sensor_w * 0.5 - cx)
    right = inv_fx_znear * (sensor_w * 0.5 + cx)
    top = inv_fy_znear * (sensor_h * 0.5 - cy)
    bottom = -inv_fy_znear * (sensor_h * 0.5 + cy)
  else:
    fovy_rad = fovy * wp.static(wp.pi / 180.0)
    half_height = znear * wp.tan(0.5 * fovy_rad)
    half_width = half_height * aspect
    left = -half_width
    right = half_width
    top = half_height
    bottom = -half_height

  u = (float(px) + 0.5) / float(img_w)
  v = (float(py) + 0.5) / float(img_h)
  x = left + (right - left) * u
  y = top + (bottom - top) * v

  return wp.normalize(wp.vec3(x, y, -znear))


@wp.func
def pack_rgba_to_uint32(r: float, g: float, b: float, a: float) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return wp.uint32((int(a) << int(24)) | (int(r) << int(16)) | (int(g) << int(8)) | int(b))

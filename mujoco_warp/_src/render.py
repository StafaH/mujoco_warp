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

from typing import Tuple

import warp as wp

from . import bvh
from . import math
from .ray import intersect_primitive
from .ray import ray_box
from .ray import ray_box_with_normal
from .ray import ray_capsule
from .ray import ray_capsule_with_normal
from .ray import ray_cylinder
from .ray import ray_cylinder_with_normal
from .ray import ray_ellipsoid
from .ray import ray_ellipsoid_with_normal
from .ray import ray_flex_with_bvh
from .ray import ray_mesh_with_bvh
from .ray import ray_plane
from .ray import ray_plane_with_normal
from .ray import ray_sphere
from .ray import ray_sphere_with_normal
from .render_context import RenderContext
from .types import Data
from .types import GeomType
from .types import Model
from .warp_util import event_scope
from .warp_util import nested_kernel

wp.set_module_options({"enable_backward": False})

MAX_NUM_VIEWS_PER_THREAD = 8

BACKGROUND_COLOR = (
  255 << 24 |
  int(0.1 * 255.0) << 16 |
  int(0.1 * 255.0) << 8 |
  int(0.2 * 255.0)
)

SPOT_INNER_COS = float(0.95)
SPOT_OUTER_COS = float(0.85)
INV_255 = float(1.0 / 255.0)
SHADOW_MIN_VISIBILITY = float(0.3)  # reduce shadow darkness (0: full black, 1: no shadow)

AMBIENT_UP = wp.vec3(0.0, 0.0, 1.0)
AMBIENT_SKY = wp.vec3(0.4, 0.4, 0.45)
AMBIENT_GROUND = wp.vec3(0.1, 0.1, 0.12)
AMBIENT_INTENSITY = float(0.5)

# Tile dimensions - can be adjusted as hyperparameters
# Must be power of 2 for optimal GPU performance
# Using wp.constant for use in tile operations
TILE_W = wp.constant(16)
TILE_H = wp.constant(16)
THREADS_PER_TILE: int = 16 * 16  # Must match TILE_W * TILE_H

# Maximum number of geometry candidates stored per tile in occupancy buffer
# If more geometries intersect, we fall back to full BVH traversal
MAX_TILE_GEOMS: int = 4

# Occupancy buffer sentinel values (stored in first element of vec4i)
# -1: No geometries intersect this tile (can skip rendering)
# -2: Too many geometries (> MAX_TILE_GEOMS), use full BVH traversal
# >= 0: First geometry ID, remaining elements contain additional IDs (-1 if unused)
OCC_EMPTY: int = -1
OCC_OVERFLOW: int = -2

# Background color as vec4 (RGBA normalized)
BACKGROUND_VEC4 = wp.vec4(0.1, 0.1, 0.2, 1.0)

@wp.func
def _ceil_div(a: int, b: int):
  return (a + b - 1) // b


# Map linear thread id (per image) -> (px, py) using TILE_W x TILE_H tiles
@wp.func
def _tile_coords(tid: int, W: int, H: int):
  tile_id = tid // THREADS_PER_TILE
  local = tid - tile_id * THREADS_PER_TILE

  u = local % TILE_W
  v = local // TILE_W

  tiles_x = _ceil_div(W, TILE_W)
  tile_x = (tile_id % tiles_x) * TILE_W
  tile_y = (tile_id // tiles_x) * TILE_H

  i = tile_x + u
  j = tile_y + v
  return i, j


@event_scope
def render(m: Model, d: Data, rc: RenderContext):
  """Render the current frame.

  Outputs are stored in buffers within the render context.

  Args:
    m: The model on device.
    d: The data on device.
    rc: The render context on device.
  """
  bvh.refit_warp_bvh(m, d, rc)
  if m.nflex:
    bvh.refit_flex_bvh(m, d, rc)
  # render_megakernel(m, d, rc)
  tile_render_megakernel(m, d, rc)

@wp.func
def pack_rgba_to_uint32(r: wp.uint8, g: wp.uint8, b: wp.uint8, a: wp.uint8) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)


@wp.func
def pack_rgba_to_uint32(r: float, g: float, b: float, a: float) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)


@wp.func
def sample_texture_2d(
  # In:
  uv: wp.vec2,
  width: int,
  height: int,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
) -> wp.vec3:
  ix = wp.min(width - 1, int(uv[0] * float(width)))
  iy = wp.min(height - 1, int(uv[1] * float(height)))
  linear_idx = tex_adr + (iy * width + ix)
  packed_rgba = tex_data[linear_idx]
  r = float((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) * INV_255
  g = float((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) * INV_255
  b = float(packed_rgba & wp.uint32(0xFF)) * INV_255
  return wp.vec3(r, g, b)


@wp.func
def sample_texture_plane(
  # In:
  hit_point: wp.vec3,
  pos: wp.vec3,
  rot: wp.mat33,
  tex_repeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  local = wp.transpose(rot) @ (hit_point - pos)
  u = local[0] * tex_repeat[0]
  v = local[1] * tex_repeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  v = 1.0 - v
  return sample_texture_2d(
    wp.vec2(u, v),
    tex_width,
    tex_height,
    tex_adr,
    tex_data,
  )


@wp.func
def sample_texture_mesh(
  # In:
  bary_u: float,
  bary_v: float,
  uv_baseadr: int,
  v_idx: wp.vec3i,
  mesh_texcoord: wp.array(dtype=wp.vec2),
  tex_repeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  bw = 1.0 - bary_u - bary_v
  uv0 = mesh_texcoord[uv_baseadr + v_idx.x]
  uv1 = mesh_texcoord[uv_baseadr + v_idx.y]
  uv2 = mesh_texcoord[uv_baseadr + v_idx.z]
  uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
  u = uv[0] * tex_repeat[0]
  v = uv[1] * tex_repeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  v = 1.0 - v
  return sample_texture_2d(
    wp.vec2(u, v),
    tex_width,
    tex_height,
    tex_adr,
    tex_data,
  )


@wp.func
def sample_texture(
  # Model:
  geom_type: wp.array(dtype=int),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  # In:
  geom_id: int,
  tex_repeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
  pos: wp.vec3,
  rot: wp.mat33,
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mesh_texcoord_offsets: wp.array(dtype=int),
  hit_point: wp.vec3,
  u: float,
  v: float,
  f: int,
  mesh_id: int,
) -> wp.vec3:
  tex_color = wp.vec3(1.0, 1.0, 1.0)

  if geom_type[geom_id] == GeomType.PLANE:
    tex_color = sample_texture_plane(
      hit_point,
      pos,
      rot,
      tex_repeat,
      tex_adr,
      tex_data,
      tex_height,
      tex_width,
    )

  if geom_type[geom_id] == GeomType.MESH:
    if f < 0 or mesh_id < 0:
      return tex_color

    base_face = mesh_faceadr[mesh_id]
    uv_base = mesh_texcoord_offsets[mesh_id]
    face_global = base_face + f
    tex_color = sample_texture_mesh(
      u,
      v,
      uv_base,
      mesh_face[face_global],
      mesh_texcoord,
      tex_repeat,
      tex_adr,
      tex_data,
      tex_height,
      tex_width,
    )

  return tex_color


@wp.func
def cast_ray(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
) -> Tuple[int, float, wp.vec3, float, float, int, int]:
  dist = float(wp.inf)
  normal = wp.vec3(0.0, 0.0, 0.0)
  geom_id = int(-1)
  bary_u = float(0.0)
  bary_v = float(0.0)
  face_idx = int(-1)
  geom_mesh_id = int(-1)

  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    # TODO: Investigate branch elimination with static loop unrolling
    if geom_type[gi] == GeomType.PLANE:
      h, d, n = ray_plane_with_normal(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.HFIELD:
      h, d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
    if geom_type[gi] == GeomType.SPHERE:
      h, d, n = ray_sphere_with_normal(
        geom_xpos_in[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.ELLIPSOID:
      h, d, n = ray_ellipsoid_with_normal(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      h, d, n = ray_capsule_with_normal(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CYLINDER:
      h, d, n = ray_cylinder_with_normal(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      h, d, n = ray_box_with_normal(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      h, d, n, u, v, f, geom_mesh_id = ray_mesh_with_bvh(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )

    if h and d < dist:
      dist = d
      normal = n
      geom_id = gi
      bary_u = u
      bary_v = v
      face_idx = f

  return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id


@wp.func
def cast_ray_first_hit(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_dist: float,
) -> bool:
  """A simpler version of cast_ray_first_hit that only checks for the first hit."""
  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, max_dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    # TODO: Investigate branch elimination with static loop unrolling
    if geom_type[gi] == GeomType.PLANE:
      d = ray_plane(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.HFIELD:
      h, d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )
    if geom_type[gi] == GeomType.SPHERE:
      d = ray_sphere(
        geom_xpos_in[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.ELLIPSOID:
      d = ray_ellipsoid(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      d = ray_capsule(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CYLINDER:
      d = ray_cylinder(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      d, all = ray_box(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      h, d, n, u, v, f, mesh_id = ray_mesh_with_bvh(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )

    if d < max_dist:
      return True

  return False


@wp.func
def compute_lighting(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),

  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),

  # In:
  use_shadows: bool,
  bvh_id: wp.uint64,
  group_root: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  world_id: int,
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  lightactive: bool,
  lighttype: int,
  lightcastshadow: bool,
  lightpos: wp.vec3,
  lightdir: wp.vec3,
  normal: wp.vec3,
  hitpoint: wp.vec3,
) -> float:

  light_contribution = float(0.0)

  # TODO: We should probably only be looping over active lights
  # in the first place with a static loop of enabled light idx?
  if not lightactive:
    return light_contribution

  L = wp.vec3(0.0, 0.0, 0.0)
  dist_to_light = float(wp.inf)
  attenuation = float(1.0)

  if lighttype == 1: # directional light
    L = wp.normalize(-lightdir)
  else:
    L, dist_to_light = math.normalize_with_norm(lightpos - hitpoint)
    attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
    if lighttype == 0: # spot light
      spot_dir = wp.normalize(lightdir)
      cos_theta = wp.dot(-L, spot_dir)
      inner = SPOT_INNER_COS
      outer = SPOT_OUTER_COS
      spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
      attenuation = attenuation * spot_factor

  ndotl = wp.max(0.0, wp.dot(normal, L))
  if ndotl == 0.0:
    return light_contribution

  visible = float(1.0)

  if use_shadows and lightcastshadow:
    # Nudge the origin slightly along the surface normal to avoid
    # self-intersection when casting shadow rays
    eps = 1.0e-4
    shadow_origin = hitpoint + normal * eps
    # Distance-limited shadows: cap by dist_to_light (for non-directional)
    max_t = float(dist_to_light - 1.0e-3)
    if lighttype == 1:  # directional light
      max_t = float(1.0e+8)

    shadow_hit = cast_ray_first_hit(
      geom_type,
      geom_dataid,
      geom_size,
      geom_xpos_in,
      geom_xmat_in,
      bvh_id,
      group_root,
      world_id,
      bvh_ngeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      shadow_origin,
      L,
      max_t,
    )

    if shadow_hit:
      visible = SHADOW_MIN_VISIBILITY

  return ndotl * attenuation * visible


@event_scope
def render_megakernel(m: Model, d: Data, rc: RenderContext):
  """Standard per-pixel rendering kernel using 4D output buffers.

  This is the non-tiled version that processes each pixel independently.
  Output format: 4D arrays (nworld, ncam, H, W) with vec4 for RGB, float for depth.
  """
  rc.rgb_data.fill_(BACKGROUND_VEC4)
  rc.depth_data.fill_(0.0)

  @nested_kernel(enable_backward="False")
  def _render_megakernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    mesh_faceadr: wp.array(dtype=int),
    mesh_face: wp.array(dtype=wp.vec3i),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    light_active: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    light_castshadow: wp.array2d(dtype=bool),

    # Data in:
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),

    # In:
    ncam: int,
    use_shadows: bool,
    bvh_ngeom: int,
    img_width: int,
    img_height: int,
    cam_id_map: wp.array(dtype=int),
    cam_ray_adr: wp.array(dtype=int),
    ray: wp.array(dtype=wp.vec3),
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    flex_bvh_id: wp.uint64,
    flex_group_root: wp.array(dtype=int),
    enabled_geom_ids: wp.array(dtype=int),
    mesh_bvh_id: wp.array(dtype=wp.uint64),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    hfield_bvh_id: wp.array(dtype=wp.uint64),
    flex_rgba: wp.array(dtype=wp.vec4),
    tex_adr: wp.array(dtype=int),
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: wp.array(dtype=int),
    tex_width: wp.array(dtype=int),

    # Out: 4D arrays (nworld, ncam, H, W)
    rgb_out: wp.array4d(dtype=wp.vec4),
    depth_out: wp.array4d(dtype=float),
  ):
    world_idx, pixel_idx = wp.tid()

    # Map pixel_idx to (cam_idx, y, x) for uniform resolution cameras
    pixels_per_cam = img_width * img_height
    cam_idx = pixel_idx // pixels_per_cam
    local_idx = pixel_idx - cam_idx * pixels_per_cam
    iy = local_idx // img_width
    ix = local_idx - iy * img_width

    if cam_idx >= ncam:
      return

    if not render_rgb[cam_idx] and not render_depth[cam_idx]:
      return

    # Map active camera index to MuJoCo camera ID
    mujoco_cam_id = cam_id_map[cam_idx]

    # Compute ray for this pixel
    ray_offset = cam_ray_adr[cam_idx]
    ray_idx_global = ray_offset + iy * img_width + ix

    ray_dir_local_cam = ray[ray_idx_global]
    ray_dir_world = cam_xmat[world_idx, mujoco_cam_id] @ ray_dir_local_cam
    ray_origin_world = cam_xpos[world_idx, mujoco_cam_id]

    geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
      geom_type,
      geom_dataid,
      geom_size,
      geom_xpos,
      geom_xmat,
      bvh_id,
      group_root[world_idx],
      world_idx,
      bvh_ngeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      ray_origin_world,
      ray_dir_world,
    )

    if wp.static(m.nflex > 0):
      h, d_flex, n_flex, u_flex, v_flex, f_flex = ray_flex_with_bvh(
        flex_bvh_id,
        flex_group_root[world_idx],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
      if h and d_flex < dist:
        dist = d_flex
        normal = n_flex
        geom_id = -2

    # Early out - no hit
    if geom_id == -1:
      return

    # Store depth
    if render_depth[cam_idx]:
      depth_out[world_idx, cam_idx, iy, ix] = dist

    if not render_rgb[cam_idx]:
      return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * dist

    if geom_id == -2:
      color = flex_rgba[0]
    elif geom_matid[world_idx, geom_id] == -1:
      color = geom_rgba[world_idx, geom_id]
    else:
      color = mat_rgba[world_idx, geom_matid[world_idx, geom_id]]

    base_color = wp.vec3(color[0], color[1], color[2])

    if wp.static(rc.use_textures):
      if geom_id != -2:
        mat_id = geom_matid[world_idx, geom_id]
        if mat_id >= 0:
          tex_id = mat_texid[world_idx, mat_id, 1]
          if tex_id >= 0:
            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              mesh_face,
              geom_id,
              mat_texrepeat[world_idx, mat_id],
              tex_adr[tex_id],
              tex_data,
              tex_height[tex_id],
              tex_width[tex_id],
              geom_xpos[world_idx, geom_id],
              geom_xmat[world_idx, geom_id],
              mesh_texcoord,
              mesh_texcoord_offsets,
              hit_point,
              u,
              v,
              f,
              mesh_id,
            )
            base_color = wp.cw_mul(base_color, tex_color)

    len_n = wp.length(normal)
    n = normal if len_n > 0.0 else AMBIENT_UP
    n = wp.normalize(n)
    hemispheric = 0.5 * (wp.dot(n, AMBIENT_UP) + 1.0)
    ambient_color = AMBIENT_SKY * hemispheric + AMBIENT_GROUND * (1.0 - hemispheric)
    result = AMBIENT_INTENSITY * wp.cw_mul(base_color, ambient_color)

    # Apply lighting and shadows
    for l in range(wp.static(m.nlight)):
      light_contribution = compute_lighting(
        geom_type,
        geom_dataid,
        geom_size,
        geom_xpos,
        geom_xmat,
        use_shadows,
        bvh_id,
        group_root[world_idx],
        bvh_ngeom,
        enabled_geom_ids,
        world_idx,
        mesh_bvh_id,
        hfield_bvh_id,
        light_active[world_idx, l],
        light_type[world_idx, l],
        light_castshadow[world_idx, l],
        light_xpos[world_idx, l],
        light_xdir[world_idx, l],
        normal,
        hit_point,
      )
      result = result + base_color * light_contribution

    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

    # Store as vec4 (RGB + alpha)
    rgb_out[world_idx, cam_idx, iy, ix] = wp.vec4(hit_color[0], hit_color[1], hit_color[2], 1.0)

  # Total pixels across all cameras
  total_pixels = rc.ncam * rc.img_width * rc.img_height

  wp.launch(
    kernel=_render_megakernel,
    dim=(d.nworld, total_pixels),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.mesh_faceadr,
      m.mesh_face,
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      m.light_active,
      m.light_type,
      m.light_castshadow,
      d.cam_xpos,
      d.cam_xmat,
      d.light_xpos,
      d.light_xdir,
      d.geom_xpos,
      d.geom_xmat,
      rc.ncam,
      rc.use_shadows,
      rc.bvh_ngeom,
      rc.img_width,
      rc.img_height,
      rc.cam_id_map,
      rc.cam_ray_adr,
      rc.ray,
      rc.render_rgb,
      rc.render_depth,
      rc.bvh_id,
      rc.group_root,
      rc.flex_bvh_id,
      rc.flex_group_root,
      rc.enabled_geom_ids,
      rc.mesh_bvh_id,
      rc.mesh_texcoord,
      rc.mesh_texcoord_offsets,
      rc.hfield_bvh_id,
      rc.flex_rgba,
      rc.tex_adr,
      rc.tex_data,
      rc.tex_height,
      rc.tex_width,
    ],
    outputs=[
      rc.rgb_data,
      rc.depth_data,
    ],
    block_dim=THREADS_PER_TILE,
  )


@event_scope
def tile_render_megakernel(m: Model, d: Data, rc: RenderContext):
  """Tile-based rendering with occupancy optimization and tile_store.

  This renderer uses a two-pass approach:
  1. Occupancy Pass: For each tile (TILE_W x TILE_H pixels), compute an AABB
     that bounds the tile's frustum and query the scene BVH. If <= MAX_TILE_GEOMS
     geometries intersect, store their IDs. Otherwise, mark as overflow.
  2. Render Pass: Using wp.launch_tiled, each block of threads handles one tile.
     After computing colors, wp.tile() gathers all per-thread colors and
     tile_store writes the 2D tile to the output buffer in one vectorized op.

  Output format: 4D arrays (nworld, ncam, H, W) with vec4 for RGB, float for depth.
  """
  # Fill with background color
  rc.rgb_data.fill_(BACKGROUND_VEC4)
  rc.depth_data.fill_(0.0)

  @nested_kernel(enable_backward="False")
  def _tile_occupancy_kernel(
    cam_res: wp.array(dtype=wp.vec2i),
    cam_id_map: wp.array(dtype=int),
    cam_ray_adr: wp.array(dtype=int),
    tile_cam_idx: wp.array(dtype=int),
    tile_top_left: wp.array(dtype=wp.vec2i),
    tile_bottom_right: wp.array(dtype=wp.vec2i),
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    ray: wp.array(dtype=wp.vec3),
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    ncam: int,
    bvh_ngeom: int,
    enabled_geom_ids: wp.array(dtype=int),
    max_t: float,
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),
    img_width: int,
    occupancy_out: wp.array2d(dtype=wp.vec4i),
  ):
    """Compute per-tile occupancy by AABB query against the scene BVH."""
    world_idx, tile_idx = wp.tid()

    cam_idx = tile_cam_idx[tile_idx]

    # Skip disabled cameras - mark as overflow to ensure they're skipped in render pass
    if not render_rgb[cam_idx] and not render_depth[cam_idx]:
      occupancy_out[world_idx, tile_idx] = wp.vec4i(OCC_OVERFLOW, -1, -1, -1)
      return

    W = img_width
    tl = tile_top_left[tile_idx]
    br = tile_bottom_right[tile_idx]
    px0 = tl[0]
    py0 = tl[1]
    px1 = br[0]
    py1 = br[1]

    ray_offset = cam_ray_adr[cam_idx]

    # Get ray directions at tile corners
    idx_tl = ray_offset + py0 * W + px0
    idx_tr = ray_offset + py0 * W + px1
    idx_bl = ray_offset + py1 * W + px0
    idx_br = ray_offset + py1 * W + px1

    mujoco_cam_id = cam_id_map[cam_idx]
    cam_origin = cam_xpos[world_idx, mujoco_cam_id]
    cam_rot = cam_xmat[world_idx, mujoco_cam_id]

    # Transform corner rays to world space
    dir_tl = cam_rot @ ray[idx_tl]
    dir_tr = cam_rot @ ray[idx_tr]
    dir_bl = cam_rot @ ray[idx_bl]
    dir_br = cam_rot @ ray[idx_br]

    # Compute near and far points along each corner ray
    # Start near plane away from origin to get a tighter AABB
    t_near = float(0.1)
    near_tl = cam_origin + dir_tl * t_near
    near_tr = cam_origin + dir_tr * t_near
    near_bl = cam_origin + dir_bl * t_near
    near_br = cam_origin + dir_br * t_near

    far_tl = cam_origin + dir_tl * max_t
    far_tr = cam_origin + dir_tr * max_t
    far_bl = cam_origin + dir_bl * max_t
    far_br = cam_origin + dir_br * max_t

    # Compute AABB that bounds the tile frustum (near + far corners)
    min_x = wp.min(wp.min(wp.min(wp.min(near_tl[0], near_tr[0]), wp.min(near_bl[0], near_br[0])),
                         wp.min(wp.min(far_tl[0], far_tr[0]), wp.min(far_bl[0], far_br[0]))),
                   cam_origin[0])
    min_y = wp.min(wp.min(wp.min(wp.min(near_tl[1], near_tr[1]), wp.min(near_bl[1], near_br[1])),
                         wp.min(wp.min(far_tl[1], far_tr[1]), wp.min(far_bl[1], far_br[1]))),
                   cam_origin[1])
    min_z = wp.min(wp.min(wp.min(wp.min(near_tl[2], near_tr[2]), wp.min(near_bl[2], near_br[2])),
                         wp.min(wp.min(far_tl[2], far_tr[2]), wp.min(far_bl[2], far_br[2]))),
                   cam_origin[2])

    max_x = wp.max(wp.max(wp.max(wp.max(near_tl[0], near_tr[0]), wp.max(near_bl[0], near_br[0])),
                         wp.max(wp.max(far_tl[0], far_tr[0]), wp.max(far_bl[0], far_br[0]))),
                   cam_origin[0])
    max_y = wp.max(wp.max(wp.max(wp.max(near_tl[1], near_tr[1]), wp.max(near_bl[1], near_br[1])),
                         wp.max(wp.max(far_tl[1], far_tr[1]), wp.max(far_bl[1], far_br[1]))),
                   cam_origin[1])
    max_z = wp.max(wp.max(wp.max(wp.max(near_tl[2], near_tr[2]), wp.max(near_bl[2], near_br[2])),
                         wp.max(wp.max(far_tl[2], far_tr[2]), wp.max(far_bl[2], far_br[2]))),
                   cam_origin[2])

    lower = wp.vec3(min_x, min_y, min_z)
    upper = wp.vec3(max_x, max_y, max_z)

    # Query BVH with the tile's AABB
    query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root[world_idx])
    bounds_nr = int(0)
    hit_count = int(0)
    g0 = int(-1)
    g1 = int(-1)
    g2 = int(-1)
    g3 = int(-1)

    while wp.bvh_query_next(query, bounds_nr):
      gi_bvh_local = bounds_nr - (world_idx * bvh_ngeom)
      if gi_bvh_local < 0 or gi_bvh_local >= bvh_ngeom:
        continue
      gid = enabled_geom_ids[gi_bvh_local]
      if hit_count == 0:
        g0 = gid
      elif hit_count == 1:
        g1 = gid
      elif hit_count == 2:
        g2 = gid
      elif hit_count == 3:
        g3 = gid
      hit_count += 1

    # Store occupancy result
    if hit_count == 0:
      # No geometries - tile can be skipped (background only)
      occupancy_out[world_idx, tile_idx] = wp.vec4i(OCC_EMPTY, -1, -1, -1)
    elif hit_count > MAX_TILE_GEOMS:
      # Too many geometries - need full BVH traversal per ray
      occupancy_out[world_idx, tile_idx] = wp.vec4i(OCC_OVERFLOW, -1, -1, -1)
    else:
      # 1-4 geometries - can use optimized direct intersection
      occupancy_out[world_idx, tile_idx] = wp.vec4i(g0, g1, g2, g3)


  @nested_kernel(enable_backward="False")
  def _tile_render_kernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    mesh_faceadr: wp.array(dtype=int),
    mesh_face: wp.array(dtype=wp.vec3i),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    light_active: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    light_castshadow: wp.array2d(dtype=bool),

    # Data in:
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),

    # In:
    ncam: int,
    use_shadows: bool,
    bvh_ngeom: int,
    img_width: int,
    img_height: int,
    cam_id_map: wp.array(dtype=int),
    cam_ray_adr: wp.array(dtype=int),
    tile_cam_idx: wp.array(dtype=int),
    tile_top_left: wp.array(dtype=wp.vec2i),
    ray: wp.array(dtype=wp.vec3),
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    flex_bvh_id: wp.uint64,
    flex_group_root: wp.array(dtype=int),
    enabled_geom_ids: wp.array(dtype=int),
    mesh_bvh_id: wp.array(dtype=wp.uint64),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    hfield_bvh_id: wp.array(dtype=wp.uint64),
    flex_rgba: wp.array(dtype=wp.vec4),
    tex_adr: wp.array(dtype=int),
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: wp.array(dtype=int),
    tex_width: wp.array(dtype=int),
    occupancy: wp.array2d(dtype=wp.vec4i),

    # Out: 4D arrays (nworld, ncam, H, W)
    rgb_out: wp.array4d(dtype=wp.vec4),
    depth_out: wp.array4d(dtype=float),
  ):
    """Render pixels using tile occupancy optimization.

    Each thread handles one pixel within a tile. Uses occupancy information
    to skip BVH traversal when only few geometries are visible in a tile.
    """
    # Global thread index: (world_idx * total_tiles * THREADS_PER_TILE + tile_idx * THREADS_PER_TILE + local_idx)
    # With block_dim=THREADS_PER_TILE, each block handles one (world, tile) combination
    global_idx = wp.tid()

    # Compute which (world, tile, local_pixel) this thread handles
    local_idx = global_idx % THREADS_PER_TILE
    block_idx = global_idx // THREADS_PER_TILE

    total_tiles = wp.static(rc.total_tiles)
    world_idx = block_idx // total_tiles
    tile_idx = block_idx % total_tiles

    # Local pixel position within the tile
    u_local = local_idx % TILE_W
    v_local = local_idx // TILE_W

    cam_idx = tile_cam_idx[tile_idx]
    tl = tile_top_left[tile_idx]

    # Pixel coordinates in image
    ix = tl[0] + u_local
    iy = tl[1] + v_local

    # Check if this thread's pixel is valid and camera is enabled
    is_valid = (ix < img_width) and (iy < img_height)
    do_rgb = render_rgb[cam_idx]
    do_depth = render_depth[cam_idx]

    if not is_valid or (not do_rgb and not do_depth):
      return

    occ = occupancy[world_idx, tile_idx]

    # Skip empty tiles - already have background color
    if occ[0] == OCC_EMPTY:
      return

    # Compute ray for this pixel
    ray_offset = cam_ray_adr[cam_idx]
    ray_idx_global = ray_offset + iy * img_width + ix

    mujoco_cam_id = cam_id_map[cam_idx]
    cam_pos = cam_xpos[world_idx, mujoco_cam_id]
    cam_rot = cam_xmat[world_idx, mujoco_cam_id]

    dir_local = ray[ray_idx_global]
    ray_dir_world = cam_rot @ dir_local
    ray_origin_world = cam_pos

    # Initialize hit result
    hit_dist = float(wp.inf)
    hit_geom = int(-1)
    hit_normal = wp.vec3(0.0, 0.0, 0.0)
    hit_u = float(0.0)
    hit_v = float(0.0)
    hit_f = int(-1)
    hit_mesh = int(-1)

    # Choose intersection strategy based on occupancy
    if occ[0] == OCC_OVERFLOW:
      # Too many geometries - use full BVH traversal
      hit_geom, hit_dist, hit_normal, hit_u, hit_v, hit_f, hit_mesh = cast_ray(
        geom_type, geom_dataid, geom_size, geom_xpos, geom_xmat,
        bvh_id, group_root[world_idx], world_idx, bvh_ngeom,
        enabled_geom_ids, mesh_bvh_id, hfield_bvh_id,
        ray_origin_world, ray_dir_world
      )
    else:
      # 1-4 geometries - directly test against each candidate
      for i in range(MAX_TILE_GEOMS):
        candidate_id = occ[i]
        if candidate_id < 0:
          break

        dist, n, u, v, f, m_id = intersect_primitive(
          candidate_id, world_idx,
          geom_type, geom_dataid, geom_size, geom_xpos, geom_xmat,
          mesh_bvh_id, hfield_bvh_id,
          ray_origin_world, ray_dir_world, hit_dist
        )

        if dist < hit_dist:
          hit_dist = dist
          hit_geom = candidate_id
          hit_normal = n
          hit_u = u
          hit_v = v
          hit_f = f
          hit_mesh = m_id

    # Also check flex objects if present
    if wp.static(m.nflex > 0):
      h_hit, d_flex, n_flex, u_flex, v_flex, f_flex = ray_flex_with_bvh(
        flex_bvh_id, flex_group_root[world_idx],
        ray_origin_world, ray_dir_world, hit_dist
      )
      if h_hit and d_flex < hit_dist:
        hit_dist = d_flex
        hit_normal = n_flex
        hit_geom = -2  # Special marker for flex hit
        hit_u = u_flex
        hit_v = v_flex
        hit_f = f_flex

    # Early exit if no hit
    if hit_geom == -1:
      return

    # Store depth
    if do_depth:
      depth_out[world_idx, cam_idx, iy, ix] = hit_dist

    if not do_rgb:
      return

    # Compute shading
    hit_point = ray_origin_world + ray_dir_world * hit_dist

    # Get base color from geometry/material
    if hit_geom == -2:
      color = flex_rgba[0]
    elif geom_matid[world_idx, hit_geom] == -1:
      color = geom_rgba[world_idx, hit_geom]
    else:
      color = mat_rgba[world_idx, geom_matid[world_idx, hit_geom]]

    base_color = wp.vec3(color[0], color[1], color[2])

    # Apply texture if enabled
    if wp.static(rc.use_textures):
      if hit_geom != -2:
        mat_id = geom_matid[world_idx, hit_geom]
        if mat_id >= 0:
          tex_id = mat_texid[world_idx, mat_id, 1]
          if tex_id >= 0:
            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              mesh_face,
              hit_geom,
              mat_texrepeat[world_idx, mat_id],
              tex_adr[tex_id],
              tex_data,
              tex_height[tex_id],
              tex_width[tex_id],
              geom_xpos[world_idx, hit_geom],
              geom_xmat[world_idx, hit_geom],
              mesh_texcoord,
              mesh_texcoord_offsets,
              hit_point,
              hit_u,
              hit_v,
              hit_f,
              hit_mesh,
            )
            base_color = wp.cw_mul(base_color, tex_color)

    # Compute ambient lighting
    len_n = wp.length(hit_normal)
    n = hit_normal if len_n > 0.0 else AMBIENT_UP
    n = wp.normalize(n)
    hemispheric = 0.5 * (wp.dot(n, AMBIENT_UP) + 1.0)
    ambient_color = AMBIENT_SKY * hemispheric + AMBIENT_GROUND * (1.0 - hemispheric)
    result = AMBIENT_INTENSITY * wp.cw_mul(base_color, ambient_color)

    # Apply lighting from each light source
    for l in range(wp.static(m.nlight)):
      light_contribution = compute_lighting(
        geom_type,
        geom_dataid,
        geom_size,
        geom_xpos,
        geom_xmat,
        use_shadows,
        bvh_id,
        group_root[world_idx],
        bvh_ngeom,
        enabled_geom_ids,
        world_idx,
        mesh_bvh_id,
        hfield_bvh_id,
        light_active[world_idx, l],
        light_type[world_idx, l],
        light_castshadow[world_idx, l],
        light_xpos[world_idx, l],
        light_xdir[world_idx, l],
        n,
        hit_point,
      )
      result = result + base_color * light_contribution

    # Clamp final color
    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

    # Store as vec4 (RGB + alpha)
    rgb_out[world_idx, cam_idx, iy, ix] = wp.vec4(hit_color[0], hit_color[1], hit_color[2], 1.0)


  # Pass 1: Compute per-tile occupancy
  occupancy = wp.zeros((d.nworld, rc.total_tiles), dtype=wp.vec4i)
  tile_max_t = 100.0  # Max ray distance for tile AABB computation

  wp.launch(
    kernel=_tile_occupancy_kernel,
    dim=(d.nworld, rc.total_tiles),
    inputs=[
      rc.cam_res,
      rc.cam_id_map,
      rc.cam_ray_adr,
      rc.tile_cam_idx,
      rc.tile_top_left,
      rc.tile_bottom_right,
      d.cam_xpos,
      d.cam_xmat,
      rc.ray,
      rc.bvh_id,
      rc.group_root,
      rc.ncam,
      rc.bvh_ngeom,
      rc.enabled_geom_ids,
      tile_max_t,
      rc.render_rgb,
      rc.render_depth,
      rc.img_width,
    ],
    outputs=[occupancy],
  )

  # Pass 2: Render pixels
  # Each thread handles one pixel. Launch with block_dim to group threads into tile blocks.
  # Total threads = nworld * total_tiles * THREADS_PER_TILE
  total_threads = d.nworld * rc.total_tiles * THREADS_PER_TILE
  wp.launch(
    kernel=_tile_render_kernel,
    dim=total_threads,
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.mesh_faceadr,
      m.mesh_face,
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      m.light_active,
      m.light_type,
      m.light_castshadow,
      d.cam_xpos,
      d.cam_xmat,
      d.light_xpos,
      d.light_xdir,
      d.geom_xpos,
      d.geom_xmat,
      rc.ncam,
      rc.use_shadows,
      rc.bvh_ngeom,
      rc.img_width,
      rc.img_height,
      rc.cam_id_map,
      rc.cam_ray_adr,
      rc.tile_cam_idx,
      rc.tile_top_left,
      rc.ray,
      rc.render_rgb,
      rc.render_depth,
      rc.bvh_id,
      rc.group_root,
      rc.flex_bvh_id,
      rc.flex_group_root,
      rc.enabled_geom_ids,
      rc.mesh_bvh_id,
      rc.mesh_texcoord,
      rc.mesh_texcoord_offsets,
      rc.hfield_bvh_id,
      rc.flex_rgba,
      rc.tex_adr,
      rc.tex_data,
      rc.tex_height,
      rc.tex_width,
      occupancy,
    ],
    outputs=[
      rc.rgb_data,
      rc.depth_data,
    ],
    block_dim=THREADS_PER_TILE,
  )

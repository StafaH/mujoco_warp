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

from typing import Tuple

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src.ray import ray_box
from mujoco_warp._src.ray import ray_capsule
from mujoco_warp._src.ray import ray_cylinder
from mujoco_warp._src.ray import ray_ellipsoid
from mujoco_warp._src.ray import ray_flex_with_bvh
from mujoco_warp._src.ray import ray_flex_with_bvh_anyhit
from mujoco_warp._src.ray import ray_mesh_with_bvh
from mujoco_warp._src.ray import ray_mesh_with_bvh_anyhit
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.ray import ray_sphere
from mujoco_warp._src.render_util import compute_ray
from mujoco_warp._src.render_util import pack_rgba_to_uint32
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def sample_texture(
  # Model:
  geom_type: wp.array(dtype=int),
  mesh_faceadr: wp.array(dtype=int),
  # In:
  geom_id: int,
  tex_repeat: wp.vec2,
  tex: wp.Texture2D,
  pos: wp.vec3,
  rot: wp.mat33,
  mesh_facetexcoord: wp.array(dtype=wp.vec3i),
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mesh_texcoord_offsets: wp.array(dtype=int),
  hit_point: wp.vec3,
  bary_u: float,
  bary_v: float,
  f: int,
  mesh_id: int,
) -> wp.vec3:
  uv = wp.vec2(0.0, 0.0)

  if geom_type[geom_id] == GeomType.PLANE:
    local = wp.transpose(rot) @ (hit_point - pos)
    uv = wp.vec2(local[0], local[1])

  if geom_type[geom_id] == GeomType.MESH:
    if f < 0 or mesh_id < 0:
      return wp.vec3(0.0, 0.0, 0.0)

    face_adr = mesh_faceadr[mesh_id] + f
    uv0 = mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][0]]
    uv1 = mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][1]]
    uv2 = mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][2]]
    uv = uv0 * bary_u + uv1 * bary_v + uv2 * (1.0 - bary_u - bary_v)

  u = uv[0] * tex_repeat[0]
  v = uv[1] * tex_repeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  tex_color = wp.texture_sample(tex, wp.vec2(u, v), dtype=wp.vec4)
  return wp.vec3(tex_color[0], tex_color[1], tex_color[2])


# TODO: Investigate combining cast_ray and cast_ray_first_hit
@wp.func
def cast_ray(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  flex_vertadr: wp.array(dtype=int),
  flex_edge: wp.array(dtype=wp.vec2i),
  flex_radius: wp.array(dtype=float),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  bvh_id: wp.uint64,
  group_root: int,
  worldid: int,
  bvh_ngeom: int,
  flex_bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  flex_geom_flexid: wp.array(dtype=int),
  flex_geom_edgeid: wp.array(dtype=int),
  flex_bvh_id: wp.array(dtype=wp.uint64),
  flex_group_root: wp.array2d(dtype=int),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
) -> Tuple[int, float, wp.vec3, float, float, int, int]:
  dist = float(MJ_MAXVAL)
  normal = wp.vec3(0.0, 0.0, 0.0)
  geom_id = int(-1)
  bary_u = float(0.0)
  bary_v = float(0.0)
  face_idx = int(-1)
  geom_mesh_id = int(-1)

  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)
  ngeom = bvh_ngeom + flex_bvh_ngeom

  while wp.bvh_query_next(query, bounds_nr, dist):
    gi_global = bounds_nr
    local_id = gi_global - (worldid * ngeom)

    d = float(-1.0)
    hit_mesh_id = int(-1)
    u = float(0.0)
    v = float(0.0)
    f = int(-1)
    n = wp.vec3(0.0, 0.0, 0.0)
    hit_geom_id = int(-1)

    if local_id < bvh_ngeom:
      gi = enabled_geom_ids[local_id]
      gtype = geom_type[gi]
    else:
      gi = local_id - bvh_ngeom
      gtype = GeomType.FLEX

    hit_geom_id = gi

    # TODO: Investigate branch elimination with static loop unrolling
    if gtype == GeomType.PLANE:
      d, n = ray_plane(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.HFIELD:
      d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
    if gtype == GeomType.SPHERE:
      d, n = ray_sphere(
        geom_xpos_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi][0] * geom_size[worldid % geom_size.shape[0], gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.ELLIPSOID:
      d, n = ray_ellipsoid(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.CAPSULE:
      d, n = ray_capsule(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.CYLINDER:
      d, n = ray_cylinder(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.BOX:
      d, all, n = ray_box(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.MESH:
      d, n, u, v, f, hit_mesh_id = ray_mesh_with_bvh(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
    if gtype == GeomType.FLEX:
      hit_geom_id = -2
      flexid = flex_geom_flexid[gi]
      edge_id = flex_geom_edgeid[gi]

      if edge_id >= 0:
        edge = flex_edge[edge_id]
        vert_adr = flex_vertadr[flexid]
        v0 = flexvert_xpos_in[worldid, vert_adr + edge[0]]
        v1 = flexvert_xpos_in[worldid, vert_adr + edge[1]]
        pos = 0.5 * (v0 + v1)
        vec = v1 - v0

        length = wp.length(vec)
        edgeq = math.quat_z2vec(vec)
        mat = math.quat_to_mat(edgeq)
        size = wp.vec3(flex_radius[flexid], 0.5 * length, 0.0)

        d, n = ray_capsule(pos, mat, size, ray_origin_world, ray_dir_world)
        hit_mesh_id = flexid
      else:
        flex_gr = flex_group_root[worldid, flexid]
        d, n, u, v, f = ray_flex_with_bvh(flex_bvh_id, flexid, flex_gr, ray_origin_world, ray_dir_world, dist)
        if d >= 0.0:
          hit_mesh_id = flexid

    if d >= 0.0 and d < dist:
      dist = d
      normal = n
      geom_id = hit_geom_id
      bary_u = u
      bary_v = v
      face_idx = f
      geom_mesh_id = hit_mesh_id

  return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id


@wp.func
def cast_ray_first_hit(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  flex_vertadr: wp.array(dtype=int),
  flex_edge: wp.array(dtype=wp.vec2i),
  flex_radius: wp.array(dtype=float),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  bvh_id: wp.uint64,
  group_root: int,
  worldid: int,
  bvh_ngeom: int,
  bvh_nflexgeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  flex_geom_flexid: wp.array(dtype=int),
  flex_geom_edgeid: wp.array(dtype=int),
  flex_bvh_id: wp.array(dtype=wp.uint64),
  flex_group_root: wp.array2d(dtype=int),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_dist: float,
) -> bool:
  """A simpler version of casting rays that only checks for the first hit."""
  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)
  ngeom = bvh_ngeom + bvh_nflexgeom

  while wp.bvh_query_next(query, bounds_nr, max_dist):
    gi_global = bounds_nr
    local_id = gi_global - (worldid * ngeom)

    d = float(-1.0)
    n = wp.vec3(0.0, 0.0, 0.0)

    if local_id < bvh_ngeom:
      gi = enabled_geom_ids[local_id]
      gtype = geom_type[gi]
    else:
      gi = local_id - bvh_ngeom
      gtype = GeomType.FLEX

    # TODO: Investigate branch elimination with static loop unrolling
    if gtype == GeomType.PLANE:
      d, n = ray_plane(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.HFIELD:
      d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )
    if gtype == GeomType.SPHERE:
      d, n = ray_sphere(
        geom_xpos_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi][0] * geom_size[worldid % geom_size.shape[0], gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.ELLIPSOID:
      d, n = ray_ellipsoid(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.CAPSULE:
      d, n = ray_capsule(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.CYLINDER:
      d, n = ray_cylinder(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.BOX:
      d, all, n = ray_box(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
    if gtype == GeomType.MESH:
      hit = ray_mesh_with_bvh_anyhit(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )
      d = 0.0 if hit else -1.0
    if gtype == GeomType.FLEX:
      flexid = flex_geom_flexid[gi]
      edge_id = flex_geom_edgeid[gi]

      if edge_id >= 0:
        edge = flex_edge[edge_id]
        vert_adr = flex_vertadr[flexid]
        v0 = flexvert_xpos_in[worldid, vert_adr + edge[0]]
        v1 = flexvert_xpos_in[worldid, vert_adr + edge[1]]
        pos = 0.5 * (v0 + v1)
        vec = v1 - v0

        length = wp.length(vec)
        edgeq = math.quat_z2vec(vec)
        mat = math.quat_to_mat(edgeq)
        size = wp.vec3(flex_radius[flexid], 0.5 * length, 0.0)

        d, n = ray_capsule(pos, mat, size, ray_origin_world, ray_dir_world)
      else:
        hit = ray_flex_with_bvh_anyhit(
          flex_bvh_id,
          flexid,
          flex_group_root[worldid, flexid],
          ray_origin_world,
          ray_dir_world,
          max_dist,
        )
        d = 0.0 if hit else -1.0

    if d >= 0.0 and d < max_dist:
      return True

  return False


@wp.func
def compute_lighting(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  flex_vertadr: wp.array(dtype=int),
  flex_edge: wp.array(dtype=wp.vec2i),
  flex_radius: wp.array(dtype=float),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  use_shadows: bool,
  bvh_id: wp.uint64,
  group_root: int,
  bvh_ngeom: int,
  bvh_nflexgeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  worldid: int,
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  flex_geom_flexid: wp.array(dtype=int),
  flex_geom_edgeid: wp.array(dtype=int),
  flex_bvh_id: wp.array(dtype=wp.uint64),
  flex_group_root: wp.array2d(dtype=int),
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
  dist_to_light = float(MJ_MAXVAL)
  attenuation = float(1.0)

  if lighttype == 1:  # directional light
    L = wp.normalize(-lightdir)
  else:
    L, dist_to_light = math.normalize_with_norm(lightpos - hitpoint)
    attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
    if lighttype == 0:  # spot light
      spot_dir = wp.normalize(lightdir)
      cos_theta = wp.dot(-L, spot_dir)
      spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - 0.85) / (0.95 - 0.85)))
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
      max_t = float(1.0e8)

    shadow_hit = cast_ray_first_hit(
      geom_type,
      geom_dataid,
      geom_size,
      flex_vertadr,
      flex_edge,
      flex_radius,
      geom_xpos_in,
      geom_xmat_in,
      flexvert_xpos_in,
      bvh_id,
      group_root,
      worldid,
      bvh_ngeom,
      bvh_nflexgeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      flex_geom_flexid,
      flex_geom_edgeid,
      flex_bvh_id,
      flex_group_root,
      shadow_origin,
      L,
      max_t,
    )

    if shadow_hit:
      visible = 0.3

  return ndotl * attenuation * visible


@event_scope
def render(m: Model, d: Data, rc: RenderContext):
  """Render the current frame.

  Outputs are stored in buffers within the render context.

  Args:
    m: The model on device.
    d: The data on device.
    rc: The render context on device.
  """
  rc.rgb_data.fill_(rc.background_color)
  rc.depth_data.fill_(0.0)
  rc.seg_data.fill_(-1)

  @wp.kernel(module="unique", enable_backward=False)
  def _render_megakernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    cam_projection: wp.array(dtype=int),
    cam_fovy: wp.array2d(dtype=float),
    cam_sensorsize: wp.array(dtype=wp.vec2),
    cam_intrinsic: wp.array2d(dtype=wp.vec4),
    light_type: wp.array2d(dtype=int),
    light_castshadow: wp.array2d(dtype=bool),
    light_active: wp.array2d(dtype=bool),
    flex_vertadr: wp.array(dtype=int),
    flex_edge: wp.array(dtype=wp.vec2i),
    flex_radius: wp.array(dtype=float),
    mesh_faceadr: wp.array(dtype=int),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    # Data in:
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    cam_xpos_in: wp.array2d(dtype=wp.vec3),
    cam_xmat_in: wp.array2d(dtype=wp.mat33),
    light_xpos_in: wp.array2d(dtype=wp.vec3),
    light_xdir_in: wp.array2d(dtype=wp.vec3),
    flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
    # In:
    nrender: int,
    use_shadows: bool,
    bvh_ngeom: int,
    bvh_nflexgeom: int,
    cam_res: wp.array(dtype=wp.vec2i),
    cam_id_map: wp.array(dtype=int),
    ray: wp.array(dtype=wp.vec3),
    rgb_adr: wp.array(dtype=int),
    depth_adr: wp.array(dtype=int),
    seg_adr: wp.array(dtype=int),
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),
    render_seg: wp.array(dtype=bool),
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    flex_bvh_id: wp.array(dtype=wp.uint64),
    flex_group_root: wp.array2d(dtype=int),
    enabled_geom_ids: wp.array(dtype=int),
    mesh_bvh_id: wp.array(dtype=wp.uint64),
    mesh_facetexcoord: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    hfield_bvh_id: wp.array(dtype=wp.uint64),
    flex_rgba: wp.array(dtype=wp.vec4),
    flex_geom_flexid: wp.array(dtype=int),
    flex_geom_edgeid: wp.array(dtype=int),
    textures: wp.array(dtype=wp.Texture2D),
    # Out:
    rgb_out: wp.array2d(dtype=wp.uint32),
    depth_out: wp.array2d(dtype=float),
    seg_out: wp.array2d(dtype=int),
  ):
    worldid, rayid = wp.tid()

    # Map global rayid -> (cam_idx, rayid_local) using cumulative sizes
    cam_idx = int(-1)
    rayid_local = int(-1)
    accum = int(0)
    for i in range(nrender):
      num_i = cam_res[i][0] * cam_res[i][1]
      if rayid < accum + num_i:
        cam_idx = i
        rayid_local = rayid - accum
        break
      accum += num_i
    if cam_idx == -1 or rayid_local < 0:
      return

    if not render_rgb[cam_idx] and not render_depth[cam_idx] and not render_seg[cam_idx]:
      return

    # Map active camera index to MuJoCo camera ID
    mujoco_cam_id = cam_id_map[cam_idx]

    if wp.static(rc.use_precomputed_rays):
      ray_dir_local_cam = ray[rayid]
    else:
      img_w = cam_res[cam_idx][0]
      img_h = cam_res[cam_idx][1]
      px = rayid_local % img_w
      py = rayid_local // img_w
      ray_dir_local_cam = compute_ray(
        cam_projection[mujoco_cam_id],
        cam_fovy[worldid % cam_fovy.shape[0], mujoco_cam_id],
        cam_sensorsize[mujoco_cam_id],
        cam_intrinsic[worldid % cam_intrinsic.shape[0], mujoco_cam_id],
        img_w,
        img_h,
        px,
        py,
        wp.static(rc.znear),
      )

    ray_dir_world = cam_xmat_in[worldid, mujoco_cam_id] @ ray_dir_local_cam
    ray_origin_world = cam_xpos_in[worldid, mujoco_cam_id]

    geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
      geom_type,
      geom_dataid,
      geom_size,
      flex_vertadr,
      flex_edge,
      flex_radius,
      geom_xpos_in,
      geom_xmat_in,
      flexvert_xpos_in,
      bvh_id,
      group_root[worldid],
      worldid,
      bvh_ngeom,
      bvh_nflexgeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      flex_geom_flexid,
      flex_geom_edgeid,
      flex_bvh_id,
      flex_group_root,
      ray_origin_world,
      ray_dir_world,
    )

    if render_seg[cam_idx] and geom_id != -1:
      seg_out[worldid, seg_adr[cam_idx] + rayid_local] = geom_id

    # Early Out
    if geom_id == -1:
      return

    if render_depth[cam_idx]:
      # Planar depth: project Euclidean distance onto the camera's optical axis.
      # In camera-local coordinates, the optical axis is -Z. The Z-component of the
      # normalized ray direction is negative, so -ray_dir_local_cam[2] gives cos(θ)
      # between the ray and the optical axis.
      depth_out[worldid, depth_adr[cam_idx] + rayid_local] = dist * (-ray_dir_local_cam[2])

    if not render_rgb[cam_idx]:
      return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * dist

    if geom_id == -2:
      # We encode flex_id in mesh_id for flex ray hits during cast_ray
      color = flex_rgba[mesh_id]
    elif geom_matid[worldid % geom_matid.shape[0], geom_id] == -1:
      color = geom_rgba[worldid % geom_rgba.shape[0], geom_id]
    else:
      color = mat_rgba[worldid % mat_rgba.shape[0], geom_matid[worldid % geom_matid.shape[0], geom_id]]

    base_color = wp.vec3(color[0], color[1], color[2])
    hit_color = base_color

    if wp.static(rc.use_textures):
      if geom_id != -2:
        mat_id = geom_matid[worldid % geom_matid.shape[0], geom_id]
        if mat_id >= 0:
          tex_id = mat_texid[worldid % mat_texid.shape[0], mat_id, 1]
          if tex_id >= 0:
            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              geom_id,
              mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],
              textures[tex_id],
              geom_xpos_in[worldid, geom_id],
              geom_xmat_in[worldid, geom_id],
              mesh_facetexcoord,
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
    n = normal if len_n > 0.0 else wp.vec3(0.0, 0.0, 1.0)
    n = wp.normalize(n)
    hemispheric = 0.5 * (n[2] + 1.0)
    ambient_color = wp.vec3(0.4, 0.4, 0.45) * hemispheric + wp.vec3(0.1, 0.1, 0.12) * (1.0 - hemispheric)
    result = 0.5 * wp.cw_mul(base_color, ambient_color)

    # Apply lighting and shadows
    for l in range(wp.static(m.nlight)):
      light_contribution = compute_lighting(
        geom_type,
        geom_dataid,
        geom_size,
        flex_vertadr,
        flex_edge,
        flex_radius,
        geom_xpos_in,
        geom_xmat_in,
        flexvert_xpos_in,
        use_shadows,
        bvh_id,
        group_root[worldid],
        bvh_ngeom,
        bvh_nflexgeom,
        enabled_geom_ids,
        worldid,
        mesh_bvh_id,
        hfield_bvh_id,
        flex_geom_flexid,
        flex_geom_edgeid,
        flex_bvh_id,
        flex_group_root,
        light_active[worldid % light_active.shape[0], l],
        light_type[worldid % light_type.shape[0], l],
        light_castshadow[worldid % light_castshadow.shape[0], l],
        light_xpos_in[worldid, l],
        light_xdir_in[worldid, l],
        normal,
        hit_point,
      )
      result = result + base_color * light_contribution

    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

    rgb_out[worldid, rgb_adr[cam_idx] + rayid_local] = pack_rgba_to_uint32(
      hit_color[0] * 255.0,
      hit_color[1] * 255.0,
      hit_color[2] * 255.0,
      255.0,
    )

  wp.launch(
    kernel=_render_megakernel,
    dim=(d.nworld, rc.total_rays),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.cam_projection,
      m.cam_fovy,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.light_type,
      m.light_castshadow,
      m.light_active,
      m.flex_vertadr,
      m.flex_edge,
      m.flex_radius,
      m.mesh_faceadr,
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      d.geom_xpos,
      d.geom_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.light_xpos,
      d.light_xdir,
      d.flexvert_xpos,
      rc.nrender,
      rc.use_shadows,
      rc.bvh_ngeom,
      rc.bvh_nflexgeom,
      rc.cam_res,
      rc.cam_id_map,
      rc.ray,
      rc.rgb_adr,
      rc.depth_adr,
      rc.seg_adr,
      rc.render_rgb,
      rc.render_depth,
      rc.render_seg,
      rc.bvh_id,
      rc.group_root,
      rc.flex_bvh_id,
      rc.flex_group_root,
      rc.enabled_geom_ids,
      rc.mesh_bvh_id,
      rc.mesh_facetexcoord,
      rc.mesh_texcoord,
      rc.mesh_texcoord_offsets,
      rc.hfield_bvh_id,
      rc.flex_rgba,
      rc.flex_geom_flexid,
      rc.flex_geom_edgeid,
      rc.textures,
    ],
    outputs=[
      rc.rgb_data,
      rc.depth_data,
      rc.seg_data,
    ],
  )


@event_scope
def raypacket_tracing_megakernel(m: Model, d: Data, rc: RenderContext):
  """Render the current frame using raypacket tracing.

  This renderer uses a two-pass approach:
  1. Occupancy Pass: Group rays into packets, using a tile block of the output
    image to create an AABB for each packet. Query the scene BVH with this AABB
    to find the geometries that intersect the packet. Store the results in an occupancy
    map, allowing for up to 4 primitives to be stored.

  2. Render Pass: Launch a tiled kernel to handle each packet. For packets <= 4 primitives, we can unroll the loop and compute the intersection directly. For packets > 4 primitives, it is more efficient to run the full BVH query for each ray again.

  Args:
    m: The model on device.
    d: The data on device.
    rc: The render context on device.
  """

  rc.rgb_data.fill_(rc.background_color)
  rc.depth_data.fill_(0.0)
  rc.seg_data.fill_(-1)

  @wp.kernel(module="unique", enable_backward=False)
  def _raypacket_occupancy_kernel(
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
    """Compute raypacket occupancy by AABB query against the scene BVH."""
    world_idx, tile_idx = wp.tid()

    cam_idx = tile_cam_idx[tile_idx]

    # Skip disabled cameras - mark as overflow to ensure they're skipped in render pass
    if not render_rgb[cam_idx] and not render_depth[cam_idx]:
      occupancy_out[world_idx, tile_idx] = wp.vec4i(-2, -1, -1, -1)
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
      occupancy_out[world_idx, tile_idx] = wp.vec4i(-1, -1, -1, -1)
    elif hit_count > 4:
      # Too many geometries - need full BVH traversal per ray
      occupancy_out[world_idx, tile_idx] = wp.vec4i(-2, -1, -1, -1)
    else:
      # 1-4 geometries - can use optimized direct intersection
      occupancy_out[world_idx, tile_idx] = wp.vec4i(g0, g1, g2, g3)


  @wp.kernel(module="unique", enable_backward=False)
  def _raypacket_render_kernel(
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
    """Render pixels using raypacket occupancy optimization.

    Each thread handles one raypacket. Uses occupancy information to skip BVH traversal
    when only few geometries are visible in a packet.
    """
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
    kernel=_raypacket_occupancy_kernel,
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

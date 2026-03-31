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
def tile_render(m: Model, d: Data, rc: RenderContext):
  """Render the current frame using raypacket tracing.

  This renderer uses a two-pass approach:
  1. Occupancy Pass: Group rays into packets, forming a bundle of rays from a square tile 
  patch of the output image to create an AABB for each packet.
  Query the scene BVH with this AABB to find the geometries that intersect the bundle.
  Store the results in an occupancy map, allowing for up to 4 primitives to be stored.

  2. Render Pass: Launch a kernel to render each packet. For packets <= 4 primitives, we can unroll the loop and compute the intersection directly. For packets > 4 primitives, it is more efficient to run the full BVH query for each ray again.
  """

  rc.rgb_out_tiled.fill_(rc.background_color)
  rc.depth_out_tiled.fill_(0.0)
  rc.seg_out_tiled.fill_(-1)

  occupancy = wp.zeros((d.nworld, rc.nrender, rc.ntiles_h * rc.ntiles_w), dtype=wp.vec4i)

  @wp.kernel(module="unique", enable_backward=False)
  def _raypacket_occupancy_kernel(
    # Data:
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    # In:
    cam_res: wp.array(dtype=wp.vec2i),
    cam_id_map: wp.array(dtype=int),
    enabled_geom_ids: wp.array(dtype=int),
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    bvh_ngeom: int,
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),
    render_seg: wp.array(dtype=bool),
    ray: wp.array(dtype=wp.vec3),
    tile_h: int,
    tile_w: int,
    ntiles_w: int,
    # Out:
    occupancy_out: wp.array3d(dtype=wp.vec4i),
  ):
    """Compute raypacket occupancy by AABB query against the scene BVH."""
    worldid, camid, tileid = wp.tid()

    # Skip disabled cameras - mark as overflow to ensure they're skipped in render pass
    if not render_rgb[camid] and not render_depth[camid] and not render_seg[camid]:
      occupancy_out[worldid, camid, tileid] = wp.vec4i(-2, -1, -1, -1)
      return

    mujoco_camid = cam_id_map[camid]

    tile_col = tileid % ntiles_w
    tile_row = tileid // ntiles_w

    # Compute pixel coordinates of tile corners, clamped to image bounds
    px = wp.vec2i(tile_col * tile_w, tile_row * tile_h)
    px1 = wp.min(px + wp.vec2i(tile_w - 1, tile_h - 1), cam_res[camid] - wp.vec2i(1, 1))

    ray_offset = camid * cam_res[camid][1] * cam_res[camid][0]

    id_tl = ray_offset + px[1] * cam_res[camid][0] + px[0]
    id_tr = ray_offset + px[1] * cam_res[camid][0] + px1[0]
    id_bl = ray_offset + px[1] * cam_res[camid][0] + px[0]
    id_br = ray_offset + px[1] * cam_res[camid][0] + px1[0]

    cam_origin = cam_xpos[worldid, mujoco_camid]
    cam_rot = cam_xmat[worldid, mujoco_camid]

    dir_tl = cam_rot @ ray[id_tl]
    dir_tr = cam_rot @ ray[id_tr]
    dir_bl = cam_rot @ ray[id_bl]
    dir_br = cam_rot @ ray[id_br]

    t_near = float(0.1)
    max_t = float(10.0)
    near_tl = cam_origin + dir_tl * t_near
    near_tr = cam_origin + dir_tr * t_near
    near_bl = cam_origin + dir_bl * t_near
    near_br = cam_origin + dir_br * t_near

    far_tl = cam_origin + dir_tl * max_t
    far_tr = cam_origin + dir_tr * max_t
    far_bl = cam_origin + dir_bl * max_t
    far_br = cam_origin + dir_br * max_t

    lower = wp.min(wp.min(wp.min(wp.min(near_tl, near_tr), wp.min(near_bl, near_br)),
                          wp.min(wp.min(far_tl, far_tr), wp.min(far_bl, far_br))),
                   cam_origin)
    upper = wp.max(wp.max(wp.max(wp.max(near_tl, near_tr), wp.max(near_bl, near_br)),
                          wp.max(wp.max(far_tl, far_tr), wp.max(far_bl, far_br))),
                   cam_origin)

    query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root[worldid])
    bounds_nr = int(0)
    hit_count = int(0)
    result = wp.vec4i(-1, -1, -1, -1)

    while wp.bvh_query_next(query, bounds_nr):
      gi_bvh_local = bounds_nr - (worldid * bvh_ngeom)
      if gi_bvh_local < 0 or gi_bvh_local >= bvh_ngeom:
        continue
      if hit_count < 4:
        result[hit_count] = enabled_geom_ids[gi_bvh_local]
      hit_count += 1

    if hit_count > 4:
      result = wp.vec4i(-2, -1, -1, -1)
    occupancy_out[worldid, camid, tileid] = result


  @wp.kernel(module="unique", enable_backward=False)
  def _raypacket_render_megakernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    flex_vertadr: wp.array(dtype=int),
    flex_edge: wp.array(dtype=wp.vec2i),
    flex_radius: wp.array(dtype=float),
    mesh_faceadr: wp.array(dtype=int),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    light_active: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    light_castshadow: wp.array2d(dtype=bool),
    # Data:
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    flexvert_xpos: wp.array2d(dtype=wp.vec3),
    # In:
    nrender: int,
    use_shadows: bool,
    bvh_ngeom: int,
    bvh_nflexgeom: int,
    cam_res: wp.array(dtype=wp.vec2i),
    cam_id_map: wp.array(dtype=int),
    ray: wp.array(dtype=wp.vec3),
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
    tile_h: int,
    tile_w: int,
    ntiles_w: int,
    total_tiles: int,
    occupancy: wp.array3d(dtype=wp.vec4i),
    # Out:
    rgb_out: wp.array4d(dtype=wp.uint32),
    depth_out: wp.array4d(dtype=float),
    seg_out: wp.array4d(dtype=int),
  ):
    """Render pixels using raypacket occupancy optimization.

    Each thread handles one pixel. Uses occupancy to choose between a fast path
    (direct intersection with 1-4 geoms) or full BVH traversal.
    """
    tid = wp.tid()

    pixels_per_tile = tile_h * tile_w
    tile_u = tid % pixels_per_tile
    tile_v = tid // pixels_per_tile

    tiles_per_world = nrender * total_tiles
    worldid = tile_v // tiles_per_world
    remainder = tile_v % tiles_per_world
    camid = remainder // total_tiles
    tileid = remainder % total_tiles

    # Local pixel position within the tile
    u_local = tile_u % tile_w
    v_local = tile_u // tile_w

    # Tile position -> pixel coordinates
    tile_col = tileid % ntiles_w
    tile_row = tileid // ntiles_w
    ix = tile_col * tile_w + u_local
    iy = tile_row * tile_h + v_local

    mujoco_cam_id = cam_id_map[camid]

    img_w = cam_res[mujoco_cam_id][0]
    img_h = cam_res[mujoco_cam_id][1]

    if ix >= img_w or iy >= img_h:
      return

    occ = occupancy[worldid, camid, tileid]

    if occ[0] == -1:
      return

    # Compute ray for this pixel
    ray_offset = camid * img_h * img_w
    ray_idx = ray_offset + iy * img_w + ix

    ray_dir_local = ray[ray_idx]
    ray_dir_world = cam_xmat[worldid, mujoco_cam_id] @ ray_dir_local
    ray_origin_world = cam_xpos[worldid, mujoco_cam_id]

    geom_id = int(-1)
    dist = float(MJ_MAXVAL)
    normal = wp.vec3(0.0, 0.0, 0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    face_idx = int(-1)
    mesh_id = int(-1)

    if occ[0] == -2:
      # Overflow - full BVH traversal
      geom_id, dist, normal, bary_u, bary_v, face_idx, mesh_id = cast_ray(
        geom_type, geom_dataid, geom_size,
        flex_vertadr, flex_edge, flex_radius,
        geom_xpos, geom_xmat, flexvert_xpos,
        bvh_id, group_root[worldid], worldid,
        bvh_ngeom, bvh_nflexgeom,
        enabled_geom_ids, mesh_bvh_id, hfield_bvh_id,
        flex_geom_flexid, flex_geom_edgeid,
        flex_bvh_id, flex_group_root,
        ray_origin_world, ray_dir_world,
      )
    else:
      # Fast path: 1-4 geoms, directly test each candidate
      for i in range(4):
        gi = occ[i]
        if gi < 0:
          break

        d = float(-1.0)
        n = wp.vec3(0.0, 0.0, 0.0)
        u = float(0.0)
        v = float(0.0)
        f = int(-1)
        m_id = int(-1)
        gtype = geom_type[gi]

        if gtype == GeomType.PLANE:
          d, n = ray_plane(
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            geom_size[worldid % geom_size.shape[0], gi],
            ray_origin_world, ray_dir_world,
          )
        if gtype == GeomType.HFIELD:
          d, n, u, v, f, _ = ray_mesh_with_bvh(
            hfield_bvh_id, geom_dataid[gi],
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            ray_origin_world, ray_dir_world, dist,
          )
        if gtype == GeomType.SPHERE:
          d, n = ray_sphere(
            geom_xpos[worldid, gi],
            geom_size[worldid % geom_size.shape[0], gi][0] * geom_size[worldid % geom_size.shape[0], gi][0],
            ray_origin_world, ray_dir_world,
          )
        if gtype == GeomType.ELLIPSOID:
          d, n = ray_ellipsoid(
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            geom_size[worldid % geom_size.shape[0], gi],
            ray_origin_world, ray_dir_world,
          )
        if gtype == GeomType.CAPSULE:
          d, n = ray_capsule(
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            geom_size[worldid % geom_size.shape[0], gi],
            ray_origin_world, ray_dir_world,
          )
        if gtype == GeomType.CYLINDER:
          d, n = ray_cylinder(
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            geom_size[worldid % geom_size.shape[0], gi],
            ray_origin_world, ray_dir_world,
          )
        if gtype == GeomType.BOX:
          d, all, n = ray_box(
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            geom_size[worldid % geom_size.shape[0], gi],
            ray_origin_world, ray_dir_world,
          )
        if gtype == GeomType.MESH:
          d, n, u, v, f, m_id = ray_mesh_with_bvh(
            mesh_bvh_id, geom_dataid[gi],
            geom_xpos[worldid, gi], geom_xmat[worldid, gi],
            ray_origin_world, ray_dir_world, dist,
          )

        if d >= 0.0 and d < dist:
          dist = d
          normal = n
          geom_id = gi
          bary_u = u
          bary_v = v
          face_idx = f
          mesh_id = m_id

    if geom_id == -1:
      return

    if render_depth[camid]:
      depth_out[worldid, camid, iy, ix] = dist * (-ray_dir_local[2])

    if render_seg[camid]:
      seg_out[worldid, camid, iy, ix] = geom_id

    if not render_rgb[camid]:
      return

    hit_point = ray_origin_world + ray_dir_world * dist

    if geom_id == -2:
      color = flex_rgba[mesh_id]
    elif geom_matid[worldid % geom_matid.shape[0], geom_id] == -1:
      color = geom_rgba[worldid % geom_rgba.shape[0], geom_id]
    else:
      color = mat_rgba[worldid % mat_rgba.shape[0], geom_matid[worldid % geom_matid.shape[0], geom_id]]

    base_color = wp.vec3(color[0], color[1], color[2])

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
              geom_xpos[worldid, geom_id],
              geom_xmat[worldid, geom_id],
              mesh_facetexcoord,
              mesh_texcoord,
              mesh_texcoord_offsets,
              hit_point,
              bary_u,
              bary_v,
              face_idx,
              mesh_id,
            )
            base_color = wp.cw_mul(base_color, tex_color)

    len_n = wp.length(normal)
    shading_n = normal if len_n > 0.0 else wp.vec3(0.0, 0.0, 1.0)
    shading_n = wp.normalize(shading_n)
    hemispheric = 0.5 * (shading_n[2] + 1.0)
    ambient_color = wp.vec3(0.4, 0.4, 0.45) * hemispheric + wp.vec3(0.1, 0.1, 0.12) * (1.0 - hemispheric)
    result = 0.5 * wp.cw_mul(base_color, ambient_color)

    for l in range(wp.static(m.nlight)):
      light_contribution = compute_lighting(
        geom_type, geom_dataid, geom_size,
        flex_vertadr, flex_edge, flex_radius,
        geom_xpos, geom_xmat, flexvert_xpos,
        use_shadows,
        bvh_id, group_root[worldid],
        bvh_ngeom, bvh_nflexgeom,
        enabled_geom_ids, worldid,
        mesh_bvh_id, hfield_bvh_id,
        flex_geom_flexid, flex_geom_edgeid,
        flex_bvh_id, flex_group_root,
        light_active[worldid % light_active.shape[0], l],
        light_type[worldid % light_type.shape[0], l],
        light_castshadow[worldid % light_castshadow.shape[0], l],
        light_xpos[worldid, l],
        light_xdir[worldid, l],
        normal,
        hit_point,
      )
      result = result + base_color * light_contribution

    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

    rgb_out[worldid, camid, iy, ix] = pack_rgba_to_uint32(
      hit_color[0] * 255.0,
      hit_color[1] * 255.0,
      hit_color[2] * 255.0,
      255.0,
    )


  wp.launch(
    kernel=_raypacket_occupancy_kernel,
    dim=(d.nworld, rc.nrender, rc.ntiles_h * rc.ntiles_w),
    inputs=[
      d.cam_xpos,
      d.cam_xmat,
      rc.cam_res,
      rc.cam_id_map,
      rc.enabled_geom_ids,
      rc.bvh_id,
      rc.group_root,
      rc.bvh_ngeom,
      rc.render_rgb,
      rc.render_depth,
      rc.render_seg,
      rc.ray,
      rc.tile_h,
      rc.tile_w,
      rc.ntiles_w,
    ],
    outputs=[occupancy],
  )

  tiles_per_cam = rc.ntiles_h * rc.ntiles_w
  pixels_per_tile = rc.tile_h * rc.tile_w
  total_threads = d.nworld * rc.nrender * tiles_per_cam * pixels_per_tile
  wp.launch(
    kernel=_raypacket_render_megakernel,
    dim=int(total_threads),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.flex_vertadr,
      m.flex_edge,
      m.flex_radius,
      m.mesh_faceadr,
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
      d.flexvert_xpos,
      rc.nrender,
      rc.use_shadows,
      rc.bvh_ngeom,
      rc.bvh_nflexgeom,
      rc.cam_res,
      rc.cam_id_map,
      rc.ray,
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
      rc.tile_h,
      rc.tile_w,
      rc.ntiles_w,
      tiles_per_cam,
      occupancy,
    ],
    outputs=[
      rc.rgb_out_tiled,
      rc.depth_out_tiled,
      rc.seg_out_tiled,
    ],
    block_dim=int(pixels_per_tile),
  )

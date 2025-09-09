from typing import Tuple

import warp as wp

from .math import safe_div
from .types import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec6


MAX_NUM_VIEWS_PER_THREAD = 8


@wp.struct
class Triangle:
  v0: wp.vec3
  v1: wp.vec3
  v2: wp.vec3


@wp.struct
class Basis:
  b0: wp.vec3
  b1: wp.vec3


@wp.func
def compute_camera_ray(
  width: int,
  height: int,
  fov_rad: float,
  px: int,
  py: int,
  cam_xpos: wp.vec3,
  cam_xmat:wp.mat33,
):
  aspect_ratio = float(width) / float(height)
  u = (float(px) + 0.5) / float(width) - 0.5
  v = (float(py) + 0.5) / float(height) - 0.5
  h = wp.tan(fov_rad / 2.0)
  ray_dir_cam_space_x = u * 2.0 * h
  ray_dir_cam_space_y = -v * 2.0 * h / aspect_ratio
  ray_dir_cam_space_z = -1.0
  ray_dir_local_cam = wp.normalize(wp.vec3(ray_dir_cam_space_x, ray_dir_cam_space_y, ray_dir_cam_space_z))
  ray_dir_world = cam_xmat @ ray_dir_local_cam
  ray_origin = cam_xpos
  return ray_dir_world, ray_origin


@wp.func
def compute_box_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  min_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  max_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        local_corner = wp.vec3(
          size[0] * (2.0 * float(i) - 1.0),
          size[1] * (2.0 * float(j) - 1.0),
          size[2] * (2.0 * float(k) - 1.0),
        )
        world_corner = pos + rot @ local_corner
        min_bound = wp.min(min_bound, world_corner)
        max_bound = wp.max(max_bound, world_corner)

  return min_bound, max_bound


@wp.func
def compute_sphere_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  return pos - wp.vec3(radius, radius, radius), pos + wp.vec3(radius, radius, radius)


@wp.func
def compute_capsule_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  half_length = size[1]
  local_end1 = wp.vec3(0.0, 0.0, -half_length)
  local_end2 = wp.vec3(0.0, 0.0, half_length)
  world_end1 = pos + rot @ local_end1
  world_end2 = pos + rot @ local_end2

  seg_min = wp.min(world_end1, world_end2)
  seg_max = wp.max(world_end1, world_end2)

  inflate = wp.vec3(radius, radius, radius)
  return seg_min - inflate, seg_max + inflate


@wp.func
def compute_plane_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  # If plane size is non-positive, treat as infinite plane and use a large default extent
  size_scale = wp.max(size[0], size[1]) * 2.0
  if size[0] <= 0.0 or size[1] <= 0.0:
    size_scale = 1000.0
  min_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  max_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  for i in range(2):
    for j in range(2):
      local_corner = wp.vec3(
        size_scale * (2.0 * float(i) - 1.0),
        size_scale * (2.0 * float(j) - 1.0),
        0.0,
      )
      world_corner = pos + rot @ local_corner
      min_bound = wp.min(min_bound, world_corner)
      max_bound = wp.max(max_bound, world_corner)

  min_bound = min_bound - wp.vec3(0.1, 0.1, 0.1)
  max_bound = max_bound + wp.vec3(0.1, 0.1, 0.1)

  return min_bound, max_bound


@wp.func
def compute_ellipsoid_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  size_scale = 1.0
  return pos - wp.vec3(size_scale, size_scale, size_scale), pos + wp.vec3(size_scale, size_scale, size_scale)


@wp.kernel
def compute_bvh_bounds(
  ngeom: int,
  nworld: int,
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_pos: wp.array2d(dtype=wp.vec3),
  geom_rot: wp.array2d(dtype=wp.mat33),
  mesh_bounds_size: wp.array(dtype=wp.vec3),
  lowers: wp.array(dtype=wp.vec3),
  uppers: wp.array(dtype=wp.vec3),
  groups: wp.array(dtype=wp.int32),
):
  tid = wp.tid()
  world_id = tid // ngeom
  geom_id = tid % ngeom

  if geom_id >= ngeom or world_id >= nworld:
    return

  pos = geom_pos[world_id, geom_id]
  rot = geom_rot[world_id, geom_id]
  size = geom_size[world_id, geom_id]
  type = geom_type[geom_id]

  if type == int(GeomType.SPHERE.value):
    lower, upper = compute_sphere_bounds(pos, rot, size)
  elif type == int(GeomType.CAPSULE.value):
    lower, upper = compute_capsule_bounds(pos, rot, size)
  elif type == int(GeomType.PLANE.value):
    lower, upper = compute_plane_bounds(pos, rot, size)
  elif type == int(GeomType.MESH.value):
    size = mesh_bounds_size[geom_dataid[geom_id]]
    lower, upper = compute_box_bounds(pos, rot, size)
  elif type == int(GeomType.ELLIPSOID.value):
    lower, upper = compute_ellipsoid_bounds(pos, rot, size)
  elif type == int(GeomType.BOX.value):
    lower, upper = compute_box_bounds(pos, rot, size)

  lowers[world_id * ngeom + geom_id] = lower
  uppers[world_id * ngeom + geom_id] = upper
  groups[world_id * ngeom + geom_id] = world_id


@wp.kernel
def compute_bvh_group_roots(
  bvh_id: wp.uint64,
  group_roots: wp.array(dtype=wp.int32),
):
  tid = wp.tid()
  root = wp.bvh_get_group_root(bvh_id, tid)
  group_roots[tid] = root


def build_warp_bvh(model: Model, data: Data):
  """Build a Warp BVH for all geometries in all worlds."""

  wp.launch(
    kernel=compute_bvh_bounds,
    dim=(data.nworld * model.ngeom),
    inputs=[
      model.ngeom,
      data.nworld,
      model.geom_type,
      model.geom_dataid,
      model.geom_size,
      data.geom_xpos,
      data.geom_xmat,
      model.mesh_bounds_size,
      data.lowers,
      data.uppers,
      data.groups,
    ],
  )

  bvh = wp.Bvh(
    data.lowers,
    data.uppers,
    groups=data.groups,
    num_groups=data.nworld,
  )

  # Store BVH handles for later queries
  data.bvh = bvh
  data.bvh_id = bvh.id

  wp.launch(
    kernel=compute_bvh_group_roots,
    dim=data.nworld,
    inputs=[bvh.id, data.group_roots],
  )


def refit_warp_bvh(model: Model, data: Data):
  wp.launch(
    kernel=compute_bvh_bounds,
    dim=(data.nworld * model.ngeom),
    inputs=[
      model.ngeom,
      data.nworld,
      model.geom_type,
      model.geom_dataid,
      model.geom_size,
      data.geom_xpos,
      data.geom_xmat,
      model.mesh_bounds_size,
      data.lowers,
      data.uppers,
      data.groups,
    ],
  )

  data.bvh.refit()


@wp.func
def intersect_single_geom(
  type: int,
  dataid: int,
  size: wp.vec3,
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  pos: wp.vec3,
  rot: wp.mat33,
  ray_o: wp.vec3,
  ray_d: wp.vec3,
  max_t: wp.float32,
) -> Tuple[bool, wp.float32, wp.vec3, wp.float32, wp.float32, int, int]:
  h = bool(False)
  d = wp.inf
  n = wp.vec3(0.0, 0.0, 0.0)
  u = wp.float32(0.0)
  v = wp.float32(0.0)
  f = int(-1)
  mesh_id = int(-1)

  if type == wp.static(GeomType.PLANE.value):
    h, d, n = ray_plane_with_normal(pos, rot, size, ray_o, ray_d)
  elif type == wp.static(GeomType.SPHERE.value):
    h, d, n = ray_sphere_with_normal(pos, size[0] * size[0], ray_o, ray_d)
  elif type == wp.static(GeomType.CAPSULE.value):
    h, d, n = ray_capsule_with_normal(pos, rot, size, ray_o, ray_d)
  elif type == wp.static(GeomType.BOX.value):
    h, d, n = ray_box_with_normal(pos, rot, size, ray_o, ray_d)
  elif type == int(GeomType.MESH.value):
    h, d, n, u, v, f, mesh_id = ray_mesh_with_normal(mesh_bvh_ids, dataid, pos, rot, ray_o, ray_d, max_t)
  
  return h, d, n, u, v, f, mesh_id


@wp.func
def _ray_map(pos: wp.vec3, mat: wp.mat33, pnt: wp.vec3, vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:
  """Maps ray to local geom frame coordinates.

  Args:
      pos: position of geom frame
      mat: orientation of geom frame
      pnt: starting point of ray in world coordinates
      vec: direction of ray in world coordinates

  Returns:
      3D point and 3D direction in local geom frame
  """

  matT = wp.transpose(mat)
  lpnt = matT @ (pnt - pos)
  lvec = matT @ vec

  return lpnt, lvec


@wp.func
def _ray_quad(a: float, b: float, c: float) -> Tuple[float, wp.vec2]:
  det = b * b - a * c
  if det < MJ_MINVAL:
    return wp.inf, wp.vec2(wp.inf, wp.inf)
  det = wp.sqrt(det)

  # compute the two solutions
  den = safe_div(1.0, a)
  x0 = (-b - det) * den
  x1 = (-b + det) * den
  x = wp.vec2(x0, x1)

  # finalize result
  if x0 >= 0.0:
    return x0, x
  elif x1 >= 0.0:
    return x1, x
  else:
    return wp.inf, x


@wp.func
def _ray_plane(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> float:
  """Returns the distance at which a ray intersects with a plane."""

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # z-vec not pointing towards front face: reject
  if lvec[2] > -MJ_MINVAL:
    return wp.inf

  # intersection with plane
  x = -lpnt[2] / lvec[2]
  if x < 0.0:
    return wp.inf

  p = wp.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])

  # accept only within rendered rectangle
  if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):
    return x
  else:
    return wp.inf


@wp.func
def ray_plane_with_normal(
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> Tuple[bool, wp.float32, wp.vec3]:
  x = _ray_plane(pos, mat, size, pnt, vec)
  if x == wp.inf:
    return False, wp.inf, wp.vec3(0.0, 0.0, 0.0)
  # Local plane normal is +Z; rotate to world space
  normal_world = mat @ wp.vec3(0.0, 0.0, 1.0)
  normal_world = wp.normalize(normal_world)
  return True, x, normal_world


@wp.func
def _ray_sphere(
  pos: wp.vec3,
  dist_sqr: float,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> wp.float32:
  """Returns the distance at which a ray intersects with a sphere."""
  dif = pnt - pos

  a = wp.dot(vec, vec)
  b = wp.dot(vec, dif)
  c = wp.dot(dif, dif) - dist_sqr

  sol, _ = _ray_quad(a, b, c)
  return sol


@wp.func
def ray_sphere_with_normal(
  pos: wp.vec3,
  dist_sqr: float,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> Tuple[bool, wp.float32, wp.vec3]:
  sol = _ray_sphere(pos, dist_sqr, pnt, vec)
  if sol == wp.inf:
    return False, wp.inf, wp.vec3(0.0, 0.0, 0.0)
  normal = wp.normalize(pnt + sol * vec - pos)
  return True, sol, normal


@wp.func
def _ray_capsule(
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> wp.float32:
  """Returns the distance at which a ray intersects with a capsule."""
  # bounding sphere test
  ssz = size[0] + size[1]
  if _ray_sphere(pos, ssz * ssz, pnt, vec) < 0.0:
    return wp.inf
  
  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # init solution
  x = -1.0

  # cylinder round side: (x * lvec + lpnt)' * (x * lvec + lpnt) = size[0] * size[0]
  sq_size0 = size[0] * size[0]
  a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
  b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
  c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - sq_size0

  # solve a * x^2 + 2 * b * x + c = 0
  sol, xx = _ray_quad(a, b, c)

  # make sure round solution is between flat sides
  if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
    if x < 0.0 or sol < x:
      x = sol

  # top cap
  ldif = wp.vec3(lpnt[0], lpnt[1], lpnt[2] - size[1])
  a += lvec[2] * lvec[2]
  b = wp.dot(lvec, ldif)
  c = wp.dot(ldif, ldif) - sq_size0
  _, xx = _ray_quad(a, b, c)

  # accept only top half of sphere
  for i in range(2):
    if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:
      if x < 0.0 or xx[i] < x:
        x = xx[i]

  # bottom cap
  ldif = wp.vec3(ldif[0], ldif[1], lpnt[2] + size[1])
  b = wp.dot(lvec, ldif)
  c = wp.dot(ldif, ldif) - sq_size0
  _, xx = _ray_quad(a, b, c)

  # accept only bottom half of sphere
  for i in range(2):
    if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:
      if x < 0.0 or xx[i] < x:
        x = xx[i]

  return x if x >= 0.0 else wp.inf


@wp.func
def ray_capsule_with_normal(
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> Tuple[bool, wp.float32, wp.vec3]:
  x = _ray_capsule(pos, mat, size, pnt, vec)
  if x == wp.inf:
    return False, wp.inf, wp.vec3(0.0, 0.0, 0.0)
  # Recompute local hit point to determine which part (side or caps) we hit
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  hit_local = lpnt + x * lvec
  # Determine normal in local space
  eps = 1.0e-6
  normal_local = wp.vec3(0.0, 0.0, 0.0)
  if wp.abs(hit_local[2]) < size[1] - eps:
    # Cylinder side: project onto XY
    normal_local = wp.normalize(wp.vec3(hit_local[0], hit_local[1], 0.0))
  elif hit_local[2] >= size[1]:
    # Top cap
    cap_center = wp.vec3(0.0, 0.0, size[1])
    normal_local = wp.normalize(hit_local - cap_center)
  else:
    # Bottom cap
    cap_center = wp.vec3(0.0, 0.0, -size[1])
    normal_local = wp.normalize(hit_local - cap_center)
  # Rotate back to world space
  normal_world = mat @ normal_local
  normal_world = wp.normalize(normal_world)
  return True, x, normal_world


_IFACE = wp.types.matrix((3, 2), dtype=int)(1, 2, 0, 2, 0, 1)


@wp.func
def _ray_box(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, vec6]:
  """Returns the distance at which a ray intersects with a box."""
  all = vec6(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

  # bounding sphere test
  ssz = wp.dot(size, size)
  if _ray_sphere(pos, ssz, pnt, vec) < 0.0:
    return wp.inf, all

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # init solution
  x = wp.inf

  # loop over axes with non-zero vec
  for i in range(3):
    if wp.abs(lvec[i]) > MJ_MINVAL:
      for side in range(-1, 2, 2):
        # solution of: lpnt[i] + x * lvec[i] = side * size[i]
        sol = (float(side) * size[i] - lpnt[i]) / lvec[i]

        # process if non-negative
        if sol >= 0.0:
          id0 = _IFACE[i][0]
          id1 = _IFACE[i][1]

          # intersection with face
          p0 = lpnt[id0] + sol * lvec[id0]
          p1 = lpnt[id1] + sol * lvec[id1]

          # accept within rectangle
          if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):
            # update
            if (x < 0.0) or (sol < x):
              x = sol

            # save in all
            all[2 * i + (side + 1) // 2] = sol

  return x, all


@wp.func
def ray_box_with_normal(
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> Tuple[bool, wp.float32, wp.vec3]:
  x, all = _ray_box(pos, mat, size, pnt, vec)
  if x == wp.inf:
    return False, wp.inf, wp.vec3(0.0, 0.0, 0.0)
  normal = wp.vec3(0.0, 0.0, 0.0)
  for i in range(3):
    if all[2 * i] != -1.0:
      normal[i] = -wp.sign(all[2 * i])
  normal_world = mat @ normal
  normal_world = wp.normalize(normal_world)
  return True, x, normal_world


@wp.func
def ray_triangle(
  triangle: Triangle,
  pnt: wp.vec3,
  vec: wp.vec3,
  basis: Basis,
) -> Tuple[wp.float32, wp.vec3]:
  dif0 = triangle.v0 - pnt
  dif1 = triangle.v1 - pnt
  dif2 = triangle.v2 - pnt

  planar_0_0 = wp.dot(dif0, basis.b0)
  planar_0_1 = wp.dot(dif0, basis.b1)
  planar_1_0 = wp.dot(dif1, basis.b0)
  planar_1_1 = wp.dot(dif1, basis.b1)
  planar_2_0 = wp.dot(dif2, basis.b0)
  planar_2_1 = wp.dot(dif2, basis.b1)

  if (
    (planar_0_0 > 0.0 and planar_1_0 > 0.0 and planar_2_0 > 0.0)
    or (planar_0_0 < 0.0 and planar_1_0 < 0.0 and planar_2_0 < 0.0)
    or (planar_0_1 > 0.0 and planar_1_1 > 0.0 and planar_2_1 > 0.0)
    or (planar_0_1 < 0.0 and planar_1_1 < 0.0 and planar_2_1 < 0.0)
  ):
    return wp.float32(wp.inf), wp.vec3(0.0, 0.0, 0.0)

  A00 = planar_0_0 - planar_2_0
  A10 = planar_1_0 - planar_2_0
  A01 = planar_0_1 - planar_2_1
  A11 = planar_1_1 - planar_2_1

  b0 = -planar_2_0
  b1 = -planar_2_1

  det = A00 * A11 - A10 * A01
  if wp.abs(det) < MJ_MINVAL:
    return wp.float32(wp.inf), wp.vec3(0.0, 0.0, 0.0)

  t0 = (A11 * b0 - A10 * b1) / det
  t1 = (-A01 * b0 + A00 * b1) / det

  if t0 < 0.0 or t1 < 0.0 or t0 + t1 > 1.0:
    return wp.float32(wp.inf), wp.vec3(0.0, 0.0, 0.0)

  dif0 = triangle.v0 - triangle.v2
  dif1 = triangle.v1 - triangle.v2
  dif2 = pnt - triangle.v2
  nrm = wp.cross(dif0, dif1)
  denom = wp.dot(vec, nrm)
  final_nrm = wp.normalize(nrm)
  if denom > 0.0:
    final_nrm = -final_nrm

  if wp.abs(denom) < MJ_MINVAL:
    return wp.float32(wp.inf), wp.vec3(0.0, 0.0, 0.0)

  dist = -wp.dot(dif2, nrm) / denom
  return wp.where(dist >= 0.0, dist, wp.float32(wp.inf)), wp.where(dist >= 0.0, final_nrm, wp.vec3(0.0, 0.0, 0.0))


@wp.func
def ray_mesh_with_normal(
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  mesh_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: wp.float32,
) -> Tuple[bool, wp.float32, wp.vec3, wp.float32, wp.float32, int, int]:
  t = wp.float32(wp.inf)
  u = wp.float32(0.0)
  v = wp.float32(0.0)
  sign = wp.float32(0.0)
  n = wp.vec3()
  f = int(-1)

  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  hit = wp.mesh_query_ray(
    mesh_bvh_ids[mesh_id], lpnt, lvec, max_t, t, u, v, sign, n, f)

  if hit and wp.dot(lvec, n) < 0.0: # Backface culling in local space
    normal = mat @ n
    normal = wp.normalize(normal)
    return True, t, normal, u, v, f, mesh_id
  
  return False, wp.inf, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1


@wp.func
def sample_texture_2d(
  uv: wp.vec2,
  width: int,
  height: int,
  nchannel: int,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint8),
) -> wp.vec3:
  ix = wp.min(width - 1, wp.int32(uv[0] * wp.float32(width)))
  iy = wp.min(height - 1, wp.int32(uv[1] * wp.float32(height)))
  base_ofs = tex_adr
  idx = base_ofs + (iy * width + ix) * nchannel
  r = wp.float32(tex_data[idx + 0]) / 255.0
  g = wp.float32(tex_data[idx + 1]) / 255.0
  b = wp.float32(tex_data[idx + 2]) / 255.0
  return wp.vec3(r, g, b)


@wp.func
def sample_texture_plane(
  hit_point: wp.vec3,
  geom_pos: wp.vec3,
  geom_rot: wp.mat33,
  mat_texrepeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint8),
  tex_nchannel: int,
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  local = wp.transpose(geom_rot) @ (hit_point - geom_pos)
  u = local[0] * mat_texrepeat[0]
  v = local[1] * mat_texrepeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  v = 1.0 - v
  return sample_texture_2d(
    wp.vec2(u, v),
    tex_width,
    tex_height,
    tex_nchannel,
    tex_adr,
    tex_data,
  )


@wp.func
def sample_texture_mesh(
  bary_u: wp.float32,
  bary_v: wp.float32,
  uv_baseadr: int,
  v_idx: wp.vec3i,
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mat_texrepeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint8),
  tex_nchannel: int,
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  bw = 1.0 - bary_u - bary_v
  uv0 = mesh_texcoord[uv_baseadr + v_idx.x]
  uv1 = mesh_texcoord[uv_baseadr + v_idx.y]
  uv2 = mesh_texcoord[uv_baseadr + v_idx.z]
  uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
  u = uv[0] * mat_texrepeat[0]
  v = uv[1] * mat_texrepeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  v = 1.0 - v
  return sample_texture_2d(
    wp.vec2(u, v),
    tex_width,
    tex_height,
    tex_nchannel,
    tex_adr,
    tex_data,
  )


@wp.func
def sample_texture(
  world_id: int,
  geom_id: int,
  geom_type: wp.array(dtype=int),
  geom_matid: int,
  mat_texid: int,
  mat_texrepeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint8),
  tex_nchannel: int,
  tex_height: int,
  tex_width: int,
  geom_xpos: wp.vec3,
  geom_xmat: wp.mat33,
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mesh_texcoord_offsets: wp.array(dtype=int),
  hit_point: wp.vec3,
  u: wp.float32,
  v: wp.float32,
  f: wp.int32,
  mesh_id: wp.int32,
) -> wp.vec3:
  tex_color = wp.vec3(1.0, 1.0, 1.0)

  if geom_matid == -1 or mat_texid == -1:
    return tex_color

  if geom_type[geom_id] == int(GeomType.PLANE.value):
    tex_color = sample_texture_plane(
      hit_point,
      geom_xpos,
      geom_xmat,
      mat_texrepeat,
      tex_adr,
      tex_data,
      tex_nchannel,
      tex_height,
      tex_width,
    )

  if geom_type[geom_id] == int(GeomType.MESH.value):
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
      mat_texrepeat,
      tex_adr,
      tex_data,
      tex_nchannel,
      tex_height,
      tex_width,
    )

  return tex_color


@wp.func
def cast_ray(
  world_id: int,
  ngeom: int,
  enabled_geom_groups_mask: wp.array(dtype=wp.uint32),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_group_mask: wp.array(dtype=wp.uint32),
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
) -> Tuple[wp.int32, wp.float32, wp.vec3, wp.float32, wp.float32, wp.int32, wp.int32]:
  dist = wp.float32(wp.inf)
  normal = wp.vec3(0.0, 0.0, 0.0)
  geom_id = wp.int32(-1)
  u = wp.float32(0.0)
  v = wp.float32(0.0)
  f = wp.int32(-1)
  mesh_id = wp.int32(-1)

  enabled_mask = enabled_geom_groups_mask[0]
  for gi in range(ngeom):
    if (geom_group_mask[gi] & enabled_mask) == 0:
      continue

    h, d, n, u, v, f, mesh_id = intersect_single_geom(
      geom_type[gi],
      geom_dataid[gi],
      geom_size[world_id, gi],
      mesh_bvh_ids,
      geom_xpos[world_id, gi],
      geom_xmat[world_id, gi],
      ray_origin_world,
      ray_dir_world,
      wp.inf,
    )
  
    if h and d < dist:
      dist = d
      normal = n
      geom_id = gi
      u = u
      v = v
      f = f
      mesh_id = mesh_id

  return geom_id, dist, normal, u, v, f, mesh_id


@wp.func
def cast_ray_bvh(
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  ngeom: int,
  enabled_geom_groups_mask: wp.array(dtype=wp.uint32),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_group_mask: wp.array(dtype=wp.uint32),
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
) -> Tuple[wp.int32, wp.float32, wp.vec3, wp.float32, wp.float32, wp.int32, wp.int32]:
  dist = wp.float32(wp.inf)
  normal = wp.vec3(0.0, 0.0, 0.0)
  geom_id = wp.int32(-1)
  u = wp.float32(0.0)
  v = wp.float32(0.0)
  f = wp.int32(-1)
  mesh_id = wp.int32(-1)

  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = wp.int32(0)

  enabled_mask = enabled_geom_groups_mask[0]
  while wp.bvh_query_next(query, bounds_nr):
    gi_global = bounds_nr
    gi = gi_global - (world_id * ngeom)

    if (geom_group_mask[gi] & enabled_mask) == 0:
      continue

    h, d, n, u, v, f, mesh_id = intersect_single_geom(
      geom_type[gi],
      geom_dataid[gi],
      geom_size[world_id, gi],
      mesh_bvh_ids,
      geom_xpos[world_id, gi],
      geom_xmat[world_id, gi],
      ray_origin_world,
      ray_dir_world,
      dist,
    )

    if h and d < dist:
      dist = d
      normal = n
      geom_id = gi
      u = u
      v = v
      f = f
      mesh_id = mesh_id

  return geom_id, dist, normal, u, v, f, mesh_id


@wp.func
def cast_ray_first_hit(
  world_id: int,
  ngeom: int,
  enabled_geom_groups_mask: wp.array(dtype=wp.uint32),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_group_mask: wp.array(dtype=wp.uint32),
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_t: wp.float32,
) -> bool:
  enabled_mask = enabled_geom_groups_mask[0]
  for gi in range(ngeom):
    if (geom_group_mask[gi] & enabled_mask) == 0:
      continue

    h, d, n, u, v, f, mesh_id = intersect_single_geom(
      geom_type[gi],
      geom_dataid[gi],
      geom_size[world_id, gi],
      mesh_bvh_ids,
      geom_xpos[world_id, gi],
      geom_xmat[world_id, gi],
      ray_origin_world,
      ray_dir_world,
      max_t,
    )

    if h and d < max_t:
      return True

  return False


@wp.func
def cast_ray_first_hit_bvh(
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  ngeom: int,
  enabled_geom_groups_mask: wp.array(dtype=wp.uint32),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_group_mask: wp.array(dtype=wp.uint32),
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_t: wp.float32,
) -> bool:
  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = wp.int32(0)

  enabled_mask = enabled_geom_groups_mask[0]

  while wp.bvh_query_next(query, bounds_nr):
    gi = bounds_nr
    gi = gi - (world_id * ngeom)

    if (geom_group_mask[gi] & enabled_mask) == 0:
      continue

    h, d, n, u, v, f, mesh_id = intersect_single_geom(
      geom_type[gi],
      geom_dataid[gi],
      geom_size[world_id, gi],
      mesh_bvh_ids,
      geom_xpos[world_id, gi],
      geom_xmat[world_id, gi],
      ray_origin_world,
      ray_dir_world,
      max_t,
    )

    if h and d < max_t:
      return True

  return False


@wp.func
def compute_lighting(
  use_shadows: bool,
  use_bvh: bool,
  bvh_id: wp.uint64,
  group_root: int,
  ngeom: int,
  enabled_geom_groups_mask: wp.array(dtype=wp.uint32),
  world_id: int,
  light_active: bool,
  light_type: int,
  light_castshadow: bool,
  light_xpos: wp.vec3,
  light_xdir: wp.vec3,
  normal: wp.vec3,
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_group_mask: wp.array(dtype=wp.uint32),
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  hit_point: wp.vec3,
) -> wp.float32:

  light_contribution = wp.float32(0.0)

  if not light_active:
    return light_contribution

  L = wp.vec3(0.0, 0.0, 0.0)
  dist_to_light = wp.float32(wp.inf)
  attenuation = wp.float32(1.0)

  if light_type == 1: # directional light
    L = wp.normalize(-light_xdir)
  else:
    to_light = light_xpos - hit_point
    dist_to_light = wp.length(to_light)
    L = wp.normalize(to_light)
    attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
    if light_type == 0: # spot light
      spot_dir = wp.normalize(light_xdir)
      cos_theta = wp.dot(-L, spot_dir)
      inner = 0.95
      outer = 0.85
      spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
      attenuation = attenuation * spot_factor

  ndotl = wp.max(0.0, wp.dot(normal, L))
  if ndotl == 0.0:
    return light_contribution

  visible = wp.float32(1.0)
  shadow_min_visibility = wp.float32(0.3) # reduce shadow darkness (0: full black, 1: no shadow)

  if use_shadows and light_castshadow:
    # Nudge the origin slightly along the surface normal to avoid
    # self-intersection when casting shadow rays
    eps = 1.0e-4
    shadow_origin = hit_point + normal * eps
    # Distance-limited shadows: cap by dist_to_light (for non-directional)
    max_t = wp.float32(dist_to_light - 1.0e-3)
    if light_type == 1:  # directional light
      max_t = wp.float32(1.0e+8)

    if use_bvh:
      shadow_hit = cast_ray_first_hit_bvh(
        bvh_id,
        group_root,
        world_id,
        ngeom,
        enabled_geom_groups_mask,
        geom_type,
        geom_dataid,
        geom_size,
        geom_group_mask,
        mesh_bvh_ids,
        geom_xpos,
        geom_xmat,
        shadow_origin,
        L,
        max_t,
      )
    else:
      shadow_hit = cast_ray_first_hit(
        world_id,
        ngeom,
        enabled_geom_groups_mask,
        geom_type,
        geom_dataid,
        geom_size,
        geom_group_mask,
        mesh_bvh_ids,
        geom_xpos,
        geom_xmat,
        shadow_origin,
        L,
        max_t,
      )

    if shadow_hit:
      visible = shadow_min_visibility

  return ndotl * attenuation * visible


@wp.kernel
def _raytrace_megakernel(
  # Model and Options
  nworld: int,
  ncam: int,
  nlight: int,
  ngeom: int,
  img_width: int,
  img_height: int,
  use_bvh: bool,
  use_textures: bool,
  use_shadows: bool,
  render_rgb: bool,
  render_depth: bool,
  
  # Camera
  fov_rad: float,
  cam_xpos: wp.array2d(dtype=wp.vec3),
  cam_xmat: wp.array2d(dtype=wp.mat33),
  
  # BVH
  bvh_id: wp.uint64,
  group_roots: wp.array(dtype=wp.int32),

  # Geometry
  enabled_geom_groups_mask: wp.array(dtype=wp.uint32),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_group_mask: wp.array(dtype=wp.uint32),
  geom_matid: wp.array2d(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  mesh_vertadr: wp.array(dtype=int),
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mesh_texcoord_offsets: wp.array(dtype=int),
  mesh_texcoord_num: wp.array(dtype=int),
  mesh_normal: wp.array(dtype=wp.vec3),
  
  # Textures
  mat_texid: wp.array3d(dtype=int),
  mat_texrepeat: wp.array2d(dtype=wp.vec2),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  tex_adr: wp.array(dtype=int),
  tex_data: wp.array(dtype=wp.uint8),
  tex_nchannel: wp.array(dtype=int),
  tex_height: wp.array(dtype=int),
  tex_width: wp.array(dtype=int),
  
  # Lights
  light_active: wp.array2d(dtype=bool),
  light_type: wp.array2d(dtype=int),
  light_castshadow: wp.array2d(dtype=bool),
  light_xpos: wp.array2d(dtype=wp.vec3),
  light_xdir: wp.array2d(dtype=wp.vec3),
  
  # Data
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  
  # Output
  out_pixels: wp.array3d(dtype=wp.vec3),
  out_depth: wp.array3d(dtype=wp.float32),
):
  tid = wp.tid()

  if tid >= nworld * ncam * img_width * img_height:
    return
  
  total_views = nworld * ncam
  pixels_per_image = img_width * img_height
  num_view_groups = (total_views + MAX_NUM_VIEWS_PER_THREAD - 1) // MAX_NUM_VIEWS_PER_THREAD

  group_idx = tid // pixels_per_image
  pixel_idx = tid % pixels_per_image

  if group_idx >= num_view_groups:
    return

  px = pixel_idx % img_width
  py = pixel_idx // img_width
  base_view = group_idx * MAX_NUM_VIEWS_PER_THREAD
  
  background_color = wp.vec3(0.1, 0.1, 0.2)
  
  for i in range(MAX_NUM_VIEWS_PER_THREAD):
    view = base_view + i
    if view >= total_views:
      break

    world_idx = view // ncam
    cam_idx = view % ncam
    
    ray_dir_world, ray_origin_world = compute_camera_ray(
      img_width,
      img_height,
      fov_rad,
      px,
      py,
      cam_xpos[world_idx, cam_idx],
      cam_xmat[world_idx, cam_idx],
    )

    if use_bvh:
      geom_id, dist, normal, u, v, f, mesh_id = cast_ray_bvh(
        bvh_id,
        group_roots[world_idx],
        world_idx,
        ngeom,
        enabled_geom_groups_mask,
        geom_type,
        geom_dataid,
        geom_size,
        geom_group_mask,
        mesh_bvh_ids,
        geom_xpos,
        geom_xmat,
        ray_origin_world,
        ray_dir_world,
      )
    else:
      geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
        world_idx,
        ngeom,
        enabled_geom_groups_mask,
        geom_type,
        geom_dataid,
        geom_size,
        geom_group_mask,
        mesh_bvh_ids,
        geom_xpos,
        geom_xmat,
        ray_origin_world,
        ray_dir_world,
      )

    # Early Out
    if geom_id == -1:
      if render_rgb:
        out_pixels[world_idx, cam_idx, pixel_idx] = background_color
      if render_depth:
        out_depth[world_idx, cam_idx, pixel_idx] = dist
      continue

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * dist

    if geom_matid[world_idx, geom_id] == -1:
      color = geom_rgba[world_idx, geom_id]
    else:
      color = mat_rgba[world_idx, geom_matid[world_idx, geom_id]]
    
    base_color = wp.vec3(color[0], color[1], color[2])
    hit_color = base_color
    
    if use_textures:
      mat_id = geom_matid[world_idx, geom_id]
      tex_id = mat_texid[world_idx, mat_id, 1]

      tex_color = sample_texture(
        world_idx,
        geom_id,
        geom_type,
        mat_id,
        tex_id,
        mat_texrepeat[world_idx, mat_id],
        tex_adr[tex_id],
        tex_data,
        tex_nchannel[tex_id],
        tex_height[tex_id],
        tex_width[tex_id],
        geom_xpos[world_idx, geom_id],
        geom_xmat[world_idx, geom_id],
        mesh_faceadr,
        mesh_face,
        mesh_texcoord,
        mesh_texcoord_offsets,
        hit_point,
        u,
        v,
        f,
        mesh_id,
      )
      base_color = wp.vec3(base_color[0] * tex_color[0], base_color[1] * tex_color[1], base_color[2] * tex_color[2])

    ambient = 0.15
    result = base_color * ambient

    # Apply lighting and shadows
    for l in range(nlight):
      light_contribution = compute_lighting(
        use_shadows,
        use_bvh,
        bvh_id,
        group_roots[world_idx],
        ngeom,
        enabled_geom_groups_mask,
        world_idx,
        light_active[world_idx, l],
        light_type[world_idx, l],
        light_castshadow[world_idx, l],
        light_xpos[world_idx, l],
        light_xdir[world_idx, l],
        normal,
        geom_type,
        geom_dataid,
        geom_size,
        geom_group_mask,
        mesh_bvh_ids,
        geom_xpos,
        geom_xmat,
        hit_point,
      )
      result = result + base_color * light_contribution

    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))
    
    if render_rgb:
      out_pixels[world_idx, cam_idx, pixel_idx] = hit_color
    if render_depth:
      out_depth[world_idx, cam_idx, pixel_idx] = dist


def render_raytrace_megakernel(model: Model, data: Data):
  total_views = data.nworld * model.ncam
  total_pixels = model.render_width * model.render_height
  num_blocks = total_views // MAX_NUM_VIEWS_PER_THREAD
  
  wp.launch(
    kernel=_raytrace_megakernel,
    dim=(num_blocks * total_pixels),
    inputs=[
      # Model and Options
      data.nworld,
      model.ncam,
      model.nlight,
      model.ngeom,
      model.render_width,
      model.render_height,
      model.use_bvh,
      model.use_textures,
      model.use_shadows,
      model.render_rgb,
      model.render_depth,

      # Camera
      model.fov_rad,
      data.cam_xpos,
      data.cam_xmat,

      # BVH
      data.bvh_id,
      data.group_roots,

      # Geometry
      model.enabled_geom_groups_mask,
      model.geom_type,
      model.geom_dataid,
      model.geom_group_mask,
      model.geom_matid,
      model.geom_size,
      model.geom_rgba,
      model.mesh_bvh_ids,
      model.mesh_faceadr,
      model.mesh_face,
      model.mesh_vertadr,
      model.mesh_texcoord,
      model.mesh_texcoord_offsets,
      model.mesh_texcoord_num,
      model.mesh_normal,

      # Textures
      model.mat_texid,
      model.mat_texrepeat,
      model.mat_rgba,
      model.tex_adr,
      model.tex_data,
      model.tex_nchannel,
      model.tex_height,
      model.tex_width,

      # Lights
      model.light_active,
      model.light_type,
      model.light_castshadow,
      data.light_xpos,
      data.light_xdir,

      # Data
      data.geom_xpos,
      data.geom_xmat,
    ],
    outputs=[
      data.pixels,
      data.depth,
    ],
  )


def render(model: Model, data: Data):
  if model.use_bvh:
    refit_warp_bvh(model, data)

  render_raytrace_megakernel(model, data)
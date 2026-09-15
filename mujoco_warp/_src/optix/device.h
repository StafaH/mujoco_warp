// Copyright 2026 The Newton Developers
// Licensed under the Apache License, Version 2.0.

struct RayHit {
  int world, geom = -1, face = -1, mesh = -1;
  float distance, u = 0, v = 0;
  wp::vec3 normal = wp::vec3(0.0f);
  bool cull, first;
};

static __forceinline__ __device__ bool ray_closer(const RayHit& h, int geom, float distance) {
  if (distance < 0.0f) return false;
  if (h.first) return distance < h.distance;
  const float tolerance = 1e-6f * wp::max(h.distance, 1.0f);
  return distance < h.distance - tolerance || (wp::abs(distance - h.distance) <= tolerance && (h.geom < 0 || geom > h.geom));
}

static __forceinline__ __device__ RayHit trace_ray(int world, wp::vec3 o, wp::vec3 d, float max_distance, bool cull,
                                                   bool first) {
  RayHit h;
  h.world = world;
  h.distance = max_distance;
  h.cull = cull;
  h.first = first;
  const auto pointer = reinterpret_cast<unsigned long long>(&h);
  unsigned low = (unsigned)pointer, high = (unsigned)(pointer >> 32);
  const int cell = world;
  const wp::vec3 offset =
      wp::vec3(static_cast<float>(cell % MJWO_GRID_WIDTH),
               static_cast<float>((cell / MJWO_GRID_WIDTH) % MJWO_GRID_WIDTH),
               static_cast<float>(cell / (MJWO_GRID_WIDTH * MJWO_GRID_WIDTH))) * MJWO_SPACING;
  o = o + offset;
  if (params.handles && params.handles[0] && max_distance > 0.0f) {
    optixTrace(params.handles[0], make_float3(o[0], o[1], o[2]), make_float3(d[0], d[1], d[2]), 0.0f, max_distance, 0.0f,
               1u << (world % 8), first ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT : OPTIX_RAY_FLAG_NONE, 0, 1, 0, low, high);
  }
  return h;
}

extern "C" __global__ void __anyhit__triangle() {
  const auto pointer = (unsigned long long)optixGetPayload_0() | ((unsigned long long)optixGetPayload_1() << 32);
  RayHit& h = *reinterpret_cast<RayHit*>(pointer);
  const int instance = optixGetInstanceId();
  if (instance / params.geom_type.shape[0] != h.world) {
    optixIgnoreIntersection();
    return;
  }
  const int geom = instance % params.geom_type.shape[0], face = optixGetPrimitiveIndex();
  const int mid = *wp::address(params.geom_dataid, h.world % params.geom_dataid.shape[0], geom);
  const bool heightfield = params.geom_type[geom] == 1;
  const auto mesh = wp::mesh_get(heightfield ? params.hfield_bvh_id[mid] : params.mesh_bvh_id[mid]);
  const auto p0 = mesh.points[mesh.indices[3 * face]], p1 = mesh.points[mesh.indices[3 * face + 1]],
             p2 = mesh.points[mesh.indices[3 * face + 2]];
  const auto mat = *wp::address(params.geom_xmat_in, h.world, geom);
  const auto normal = wp::mul(mat, wp::normalize(wp::cross(p1 - p0, p2 - p0)));
  const auto direction = optixGetWorldRayDirection();
  // Shadow meshes are two-sided.
  if (h.cull && wp::dot(normal, wp::vec3(direction.x, direction.y, direction.z)) >= 0.0f) {
    optixIgnoreIntersection();
    return;
  }
  const float distance = optixGetRayTmax();
  if (ray_closer(h, geom, distance)) {
    const float2 bary = optixGetTriangleBarycentrics();
    h.geom = geom;
    h.distance = distance;
    h.normal = normal;
    h.u = 1.0f - bary.x - bary.y;
    h.v = bary.x;
    h.face = face;
    h.mesh = heightfield ? -1 : mid;
    if (h.first) {
      optixTerminateRay();
      return;
    }
  }
  // Keep near-coincident candidates available for the shared geom-ID tie break.
  optixIgnoreIntersection();
}

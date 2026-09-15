# Copyright 2026 The Newton Developers
# Licensed under the Apache License, Version 2.0.
"""Probe warp_optix graph replay in a separate process; requires the optional upstream addon.

Verifies eager output first, then checks repeated replay with changing ray origins.
A capture that returns a graph but omits rendering work must fail this probe.
Use a separate process because unsupported driver capture can invalidate CUDA.
"""

import numpy as np
import warp as wp
import warp_optix as wo


@wp.struct
class Params:
  """Persistent launch bindings shared by eager and captured submissions."""

  output: wp.array(dtype=wp.int32)
  origins: wp.array(dtype=wp.vec3)
  traversable: wp.uint64


@wp.struct
class Payload:
  """One payload register required by the OptiX trace builtin."""

  value: wp.uint32


@wo.optix_kernel(wo.OptixKernelType.RAYGEN)
def raygen(p: Params):
  i = wp.optix_get_launch_index()[0]
  payload = Payload()
  payload.value = wp.uint32(0)
  wp.optix_trace(
    p.traversable,
    p.origins[i],
    wp.vec3(0.0, 0.0, -1.0),
    0.0,
    100.0,
    0.0,
    wp.uint32(255),
    wp.uint32(0),
    wp.uint32(0),
    wp.uint32(1),
    wp.uint32(0),
    payload,
  )


@wo.optix_kernel(wo.OptixKernelType.MISS)
def miss(p: Params):
  p.output[wp.optix_get_launch_index()[0]] = -1


@wo.optix_kernel(wo.OptixKernelType.CLOSEST_HIT)
def closest(p: Params):
  p.output[wp.optix_get_launch_index()[0]] = 1


def main():
  wp.init()
  optix = wo.require_optix()
  print("OptiX", optix.version(), flush=True)
  stream = wp.Stream()
  with wp.ScopedStream(stream):
    ctx, logger = wo.create_context(optix, int(wp.get_device().context))
    ptx = wo.compile_warp_module_to_ptx(wp.get_module(__name__), "", "capture_probe", __file__)
    gas, resources = wo.create_triangle_gas(
      optix,
      ctx,
      np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32),
      np.array([[0, 1, 2]], dtype=np.uint32),
      "cuda",
    )
    pipeline, sbt, buffers = wo.create_pipeline_and_sbt(optix, ctx, ptx, raygen, miss, closest, 1, 2, "cuda")
    p = Params()
    p.output = wp.zeros(32, dtype=wp.int32)
    p.origins = wp.array(np.tile([0, 0, 2], (32, 1)), dtype=wp.vec3)
    p.traversable = gas
    params = wo.create_launch_params_buffer(Params, "cuda")
    wo.write_launch_params(params, p)

    def launch():
      wo.launch(optix, pipeline, sbt, 32, 1, params, stream=stream.cuda_stream)

    launch()
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(p.output.numpy(), 1)
    print("Eager output verified", flush=True)
    with wp.ScopedCapture(stream=stream) as capture:
      launch()
    for i in range(10):
      p.origins.assign(np.tile([0 if i % 2 == 0 else 3, 0, 2], (32, 1)))
      p.output.zero_()
      wp.capture_launch(capture.graph, stream=stream)
      wp.synchronize_stream(stream)
      np.testing.assert_array_equal(p.output.numpy(), 1 if i % 2 == 0 else -1)
      print("replay verified", i, flush=True)


if __name__ == "__main__":
  main()

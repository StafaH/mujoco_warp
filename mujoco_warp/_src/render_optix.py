# Copyright 2026 The Newton Developers
# Licensed under the Apache License, Version 2.0.
"""OptiX launch adapter for the Warp batch renderer.

The adapter deliberately contains all dependencies on Warp's code generation ABI.
Ordinary CUDA kernels cannot call optixTrace: the shared renderer is compiled as
an OptiX ray-generation program. One spatially packed instance hierarchy serves
the batch, while Warp supplies the shared analytic and flex intersections.
"""

import ctypes
import functools
import hashlib
import os
import shutil
import subprocess
import weakref
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
from warp._src import codegen
from warp._src import context

from mujoco_warp._src.types import GeomType

wp.set_module_options({"enable_backward": False})


def _paths():
  sdk = Path(os.environ.get("OPTIX_PATH", "/opt/optix"))
  include = sdk if (sdk / "optix.h").is_file() else sdk / "include"
  cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
  if not (include / "optix.h").is_file():
    raise RuntimeError("OptiX requires SDK headers. Set OPTIX_PATH to an NVIDIA OptiX 9.0 SDK directory.")
  if "#define OPTIX_VERSION 90000" not in (include / "optix.h").read_text():
    raise RuntimeError("This adapter requires OptiX SDK 9.0 headers. Set OPTIX_PATH to the v9.0.0 SDK.")
  if not (cuda / "include/cuda.h").is_file() or not shutil.which("c++"):
    raise RuntimeError("OptiX requires the CUDA toolkit (CUDA_HOME) and a C++ compiler.")
  return include, cuda


def check_available():
  """Check prerequisites without affecting the default software backend."""
  if not wp.get_device().is_cuda:
    raise ValueError("backend='optix' requires an NVIDIA CUDA device")
  if wp.__version__.split(".")[:2] != ["1", "15"]:
    raise RuntimeError("The OptiX codegen adapter requires Warp 1.15.x; install mujoco-warp[optix].")
  _paths()


@functools.lru_cache(maxsize=1)
def _bridge():
  include, cuda = _paths()
  source = Path(__file__).with_name("optix") / "bridge.cpp"
  digest = hashlib.sha256(source.read_bytes() + (include / "optix.h").read_bytes()).hexdigest()[:16]
  cache = Path(wp.config.kernel_cache_dir) / "mjwarp_optix"
  cache.mkdir(parents=True, exist_ok=True)
  library = cache / f"bridge_{digest}.so"
  if not library.exists():
    temporary = library.with_suffix(f".{os.getpid()}.so")
    subprocess.run(
      [
        "c++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        str(source),
        f"-I{include}",
        f"-I{cuda / 'include'}",
        f"-L{cuda / 'lib64/stubs'}",
        "-lcuda",
        "-ldl",
        "-o",
        str(temporary),
      ],
      check=True,
    )
    temporary.replace(library)
  lib = ctypes.CDLL(str(library))
  ptr, u64, size, integer = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int
  signatures = {
    "error": (ctypes.c_char_p, []),
    "create": (ptr, [ptr]),
    "handles": (u64, [ptr]),
    "instances": (integer, [ptr, ptr, u64, ctypes.c_uint]),
    "build_scene": (integer, [ptr, ptr]),
    "mesh": (u64, [ptr, ptr, u64, ctypes.c_uint, u64, ctypes.c_uint]),
    "compile": (integer, [ptr, ctypes.c_char_p, size]),
    "launch": (integer, [ptr, ptr, u64, size, ctypes.c_uint, ctypes.c_uint]),
    "destroy": (None, [ptr]),
  }
  for name, (restype, argtypes) in signatures.items():
    fn = getattr(lib, f"mjwo_{name}")
    fn.restype, fn.argtypes = restype, argtypes
  return lib


def _destroy_renderer(device, lib, handle):
  # Finalizers run before Python tears down Warp's module globals at shutdown.
  with device.context_guard:
    wp.synchronize_device(device)
    lib.mjwo_destroy(handle)


@wp.kernel
def _write_instances(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  geom_ids: wp.array[int],
  mesh_handles: wp.array[wp.uint64],
  hfield_handles: wp.array[wp.uint64],
  grid_width: int,
  spacing: float,
  # Out:
  instances_out: wp.array3d[wp.uint32],
  transforms_out: wp.array3d[float],
):
  world, index = wp.tid()
  geom = geom_ids[index]
  mid = geom_dataid[world % geom_dataid.shape[0], geom]
  handle = wp.uint64(0)
  if geom_type[geom] == GeomType.MESH:
    handle = mesh_handles[mid]
  else:
    handle = hfield_handles[mid]
  cell = world
  offset = (
    wp.vec3(float(cell % grid_width), float((cell // grid_width) % grid_width), float(cell // (grid_width * grid_width)))
    * spacing
  )
  pos = geom_xpos_in[world % geom_xpos_in.shape[0], geom] + offset
  mat = geom_xmat_in[world % geom_xmat_in.shape[0], geom]
  for row in range(3):
    for col in range(3):
      transforms_out[world, index, row * 4 + col] = mat[row, col]
    transforms_out[world, index, row * 4 + 3] = pos[row]
  instances_out[world, index, 12] = wp.uint32(world * geom_type.shape[0] + geom)
  instances_out[world, index, 13] = wp.uint32(0)
  instances_out[world, index, 14] = wp.uint32(1 << (world % 8))
  instances_out[world, index, 15] = wp.uint32(0)
  instances_out[world, index, 16] = wp.uint32(handle & wp.uint64(0xFFFFFFFF))
  instances_out[world, index, 17] = wp.uint32(handle >> wp.uint64(32))


class OptixRenderer:
  """Own the OptiX context, acceleration structures, pipeline, and launch buffers."""

  def __init__(self, mjm, rc, nworld):
    self.device = rc.rgb_data.device
    self.lib = _bridge()
    self.handle = None
    self.rc = weakref.proxy(rc)
    self.params = {}
    self.kernel = None
    self.cast_functions = ()
    self.grid_width = max(1, int(np.ceil(nworld ** (1.0 / 3.0))))
    self.spacing = float(mjm.stat.extent) * 2.0
    enabled = rc.enabled_geom_ids.numpy()
    geoms = enabled[np.isin(mjm.geom_type[enabled], [int(GeomType.MESH), int(GeomType.HFIELD)])]
    if len(geoms) and nworld * mjm.ngeom > (1 << 24):
      raise ValueError("OptiX batch exceeds the 24-bit instance ID limit")
    self.geom_ids = wp.array(geoms, dtype=int, device=self.device)
    self.analytic_ids = tuple(int(i) for i, g in enumerate(enabled) if g not in geoms)
    self.analytic_ids += tuple(range(rc.bvh_ngeom, rc.bvh_ngeom + rc.bvh_nflexgeom))
    # OptixInstance is 80 bytes: 12 transform floats, four control words,
    # a 64-bit traversable handle, and two padding words. The bridge checks this ABI.
    self.instances = wp.zeros((nworld, len(geoms), 20), dtype=wp.uint32, device=self.device)
    with wp.ScopedDevice(self.device):
      self.handle = self.lib.mjwo_create(self.device.context)
      if not self.handle:
        raise RuntimeError(self.lib.mjwo_error().decode())
      self._finalizer = weakref.finalize(self, _destroy_renderer, self.device, self.lib, self.handle)

      def build_meshes(ids, registry):
        handles = []
        for mid in ids.numpy():
          if not mid:
            handles.append(0)
            continue
          mesh = registry[int(mid)]
          handle = self.lib.mjwo_mesh(
            self.handle,
            wp.get_stream().cuda_stream,
            mesh.points.ptr,
            len(mesh.points),
            mesh.indices.ptr,
            len(mesh.indices) // 3,
          )
          if not handle:
            raise RuntimeError(self.lib.mjwo_error().decode())
          handles.append(handle)
        return wp.array(handles, dtype=wp.uint64, device=self.device)

      self.mesh_handles = build_meshes(rc.mesh_bvh_id, rc.mesh_registry)
      self.hfield_handles = build_meshes(rc.hfield_bvh_id, rc.hfield_registry)

      if len(geoms):
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        self._instances(
          wp.array(mjm.geom_type, dtype=int),
          wp.array(mjm.geom_dataid[None], dtype=int),
          wp.array(mjd.geom_xpos[None], dtype=wp.vec3),
          wp.array(mjd.geom_xmat.reshape(1, -1, 3, 3), dtype=wp.mat33),
        )
        self._check(self.lib.mjwo_instances(self.handle, wp.get_stream().cuda_stream, self.instances.ptr, nworld * len(geoms)))

  def _instances(self, types, ids, xpos, xmat):
    wp.launch(
      _write_instances,
      self.instances.shape[:2],
      inputs=[types, ids, xpos, xmat, self.geom_ids, self.mesh_handles, self.hfield_handles, self.grid_width, self.spacing],
      outputs=[self.instances, self.instances.view(float)],
      device=self.device,
    )

  def refit(self, m, d):
    """Pack current transforms and mesh selections; this stage supports graph capture."""
    if self.geom_ids.size:
      self._instances(m.geom_type, m.geom_dataid, d.geom_xpos, d.geom_xmat)

  def prepare(self):
    """Build the batch mesh hierarchy once per render, after any captured simulation."""
    if self.geom_ids.size:
      self.check_capture()
      with self.device.context_guard:
        self._check(self.lib.mjwo_build_scene(self.handle, wp.get_stream(self.device).cuda_stream))

  def _check(self, result):
    if result:
      raise RuntimeError(self.lib.mjwo_error().decode())

  def check_capture(self):
    """Reject unsupported OptiX capture before calling the driver."""
    if self.geom_ids.size and wp.get_stream(self.device).is_capturing:
      raise RuntimeError("OptiX rendering cannot be captured in a CUDA graph. Render after simulation graph replay.")

  def _compile(self, kernel, values):
    from mujoco_warp._src.render import _make_cast_ray

    if wp.get_stream(self.device).is_capturing:
      raise RuntimeError("Warm up mjw.render before capturing an OptiX render in a CUDA graph")
    options = kernel.module.options | {"block_dim": 128, "enable_backward": False}
    builder = context.ModuleBuilder(kernel.module, options)
    analytic_types = tuple(t for t in self.rc.geom_ray_types if t not in (int(GeomType.MESH), int(GeomType.HFIELD)))
    local_ids = self.analytic_ids if len(self.analytic_ids) <= 32 else None
    analytic_casts = [_make_cast_ray(analytic_types, first, local_ids) for first in (False, True)]
    for func in analytic_casts:
      builder.build_function(func)
    packed = [context.pack_arg(kernel, a.type, a.label, v, self.device) for a, v in zip(kernel.adj.args, values, strict=True)]
    self.param_type = type(
      "LaunchParams",
      (ctypes.Structure,),
      {"_fields_": [(a.label, type(v)) for a, v in zip(kernel.adj.args, packed, strict=True)] + [("handles", ctypes.c_uint64)]},
    )
    fields = "\n".join(f"  {a.ctype()} {a.label};" for a in kernel.adj.args)
    header = codegen.cuda_module_header.format(block_dim=128)
    header = "#include <optix.h>\n#include <optix_device.h>\n" + header
    header += f"\nstruct LaunchParams {{\n{fields}\n  const unsigned long long* handles;\n}};\n"
    header += f'static_assert(sizeof(LaunchParams) == {ctypes.sizeof(self.param_type)}, "Launch parameter ABI mismatch");\n'
    header += f'extern "C" {{ __constant__ __align__(16) unsigned char mjwo_params[{ctypes.sizeof(self.param_type)}]; }}\n'
    header += "#define params (*reinterpret_cast<const LaunchParams*>(mjwo_params))\n"
    header += f"#define MJWO_GRID_WIDTH {self.grid_width}\n#define MJWO_SPACING {self.spacing:.9e}f\n"
    header += (Path(__file__).with_name("optix") / "device.h").read_text()
    source = header
    for func in analytic_casts:
      args = [f"{a.ctype()} var_{a.label}" for a in func.adj.args]
      args += [f"{a.ctype()}& ret_{i}" for i, a in enumerate(func.adj.return_var)]
      source += f"static CUDA_CALLABLE void {func.native_func}({', '.join(args)});\n"
    for function in builder.functions:
      closure = dict(zip(function.func.__code__.co_freevars, function.func.__closure__ or ()))
      cast_types = None
      if function.key == self.cast_functions[0].key:
        primitive_func = closure["ray_primitive"].cell_contents.func
        primitive_closure = dict(zip(primitive_func.__code__.co_freevars, primitive_func.__closure__))
        cast_types = primitive_closure["geom_ray_types"].cell_contents
      if cast_types == self.rc.geom_ray_types:
        first = closure["first_hit"].cell_contents
        args = [f"{a.ctype()} var_{a.label}" for a in function.adj.args]
        args += [f"{a.ctype()}& ret_{i}" for i, a in enumerate(function.adj.return_var)]
        source += f"\nstatic CUDA_CALLABLE void {function.native_func}({', '.join(args)}) {{\n"
        source += (
          "  const auto h = trace_ray(var_worldid, var_ray_origin_world, var_ray_dir_world, "
          f"var_max_dist, var_cull_backfaces, {str(first).lower()});\n"
        )
        if first:
          source += (
            "  if (h.geom >= 0) {\n"
            "    ret_0=h.geom; ret_1=h.distance; ret_2=h.normal; ret_3=h.u; ret_4=h.v; ret_5=h.face; ret_6=h.mesh;\n"
            "    return;\n  }\n"
          )
        calls = [
          f"var_{a.label}"
          if a.label != "max_dist"
          else ("var_max_dist" if first else "wp::min(var_max_dist, h.distance * (1.0f + 1e-6f) + 1e-9f)")
          for a in function.adj.args
        ]
        calls += [f"ret_{i}" for i in range(7)]
        source += f"  {analytic_casts[int(first)].native_func}({', '.join(calls)});\n"
        source += (
          "  const float tol = 1e-6f * wp::max(h.distance, 1.0f);\n"
          "  if (h.geom >= 0 && (ret_0 == -1 || h.distance < ret_1 - tol || "
          "(wp::abs(h.distance-ret_1) <= tol && h.geom > ret_0))) {\n"
          "    ret_0=h.geom; ret_1=h.distance; ret_2=h.normal; ret_3=h.u; ret_4=h.v; ret_5=h.face; ret_6=h.mesh;\n  }\n}\n"
        )
      else:
        source += builder._codegen_functions([function], "cuda", forward_only=True)
    # Warp's forward body has no CUDA-specific launch code; wp.tid() is mapped
    # through the regular launch bounds to OptiX's two-dimensional launch index.
    body = codegen.codegen_func_forward(kernel.adj, func_type="kernel", device="cuda", grid_stride=False)
    aliases = "\n".join(f"  auto var_{a.label} = params.{a.label};" for a in kernel.adj.args)
    source += (
      """\nextern "C" __global__ void __raygen__render() {
  const auto index = optixGetLaunchIndex();
  const auto dimensions = optixGetLaunchDimensions();
  wp::launch_bounds_t<2> dim;
  dim.shape[0] = dimensions.y; dim.shape[1] = dimensions.x;
  dim.size = (size_t)dimensions.x * dimensions.y;
  const size_t _idx = (size_t)index.y * dimensions.x + index.x;
"""
      + aliases
      + "\n"
      + body
      + "\n}\n"
    )
    include, _ = _paths()
    native = Path(wp.__file__).parent / "native"
    # OptiX has no CUDA block/shared-memory model. Use Warp's existing local
    # stack for grouped scene traversal, flexes, and splat compositing.
    original_bvh = (native / "bvh.h").read_text()
    shared_stack = "#ifdef __CUDA_ARCH__\n#define BVH_SHARED_STACK 1"
    if shared_stack not in original_bvh:
      raise RuntimeError("This Warp version changed the BVH stack ABI; update the OptiX codegen adapter")
    header_digest = hashlib.sha256(original_bvh.encode()).hexdigest()[:12]
    adapted = Path(wp.config.kernel_cache_dir) / "mjwarp_optix" / f"native_{header_digest}"
    if not adapted.exists():
      temporary = adapted.with_name(adapted.name + f"_{os.getpid()}")
      shutil.copytree(native, temporary, dirs_exist_ok=True)
      (temporary / "bvh.h").write_text(original_bvh.replace(shared_stack, "#if 0\n#define BVH_SHARED_STACK 1", 1))
      try:
        temporary.rename(adapted)
      except OSError:
        if not adapted.is_dir():
          raise
        shutil.rmtree(temporary)
    digest = hashlib.sha256(
      (source + header_digest + wp.__version__ + str(self.device.arch) + str(self.rc.use_fast_math)).encode()
    ).hexdigest()[:20]
    cache = Path(wp.config.kernel_cache_dir) / "mjwarp_optix"
    cu, ptx = cache / f"render_{digest}.cu", cache / f"render_{digest}.ptx"
    if not ptx.exists():
      cu.write_text(source)
      temporary_ptx = ptx.with_name(ptx.stem + f"_{os.getpid()}.ptx")
      includes = (ctypes.c_char_p * 1)(str(include).encode())
      result = context.runtime.core.wp_cuda_compile_program(
        source.encode(),
        cu.name.encode(),
        self.device.arch,
        b"",
        str(adapted).encode(),
        1,
        includes,
        False,
        3,
        False,
        False,
        self.rc.use_fast_math,
        True,
        False,
        False,
        False,
        str(temporary_ptx).encode(),
        None,
        0,
        None,
        None,
        None,
      )
      if result:
        raise RuntimeError(f"OptiX device compilation failed ({cu})")
      temporary_ptx.replace(ptx)
    program = ptx.read_bytes()
    with self.device.context_guard:
      self._check(self.lib.mjwo_compile(self.handle, program, len(program)))
    self.kernel = kernel

  def launch(self, kernel, dim, inputs, outputs, block_dim):
    """Launch shared shading through OptiX on the current Warp stream."""
    if not self.geom_ids.size:
      wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, block_dim=block_dim, device=self.device)
      return
    self.check_capture()
    if dim[0] == 0 or dim[1] == 0:
      return
    values = inputs + outputs
    if self.kernel is None:
      self._compile(kernel, values)
    packed = [context.pack_arg(kernel, a.type, a.label, v, self.device) for a, v in zip(kernel.adj.args, values, strict=True)]
    params = self.param_type(*packed, self.lib.mjwo_handles(self.handle))
    contents = bytes(params)
    sample = inputs[[a.label for a in kernel.adj.args].index("ray_base")]
    previous = self.params.get(sample)
    if previous is None or previous[0] != contents:
      buffer = wp.array(np.frombuffer(contents, dtype=np.uint8).copy(), dtype=wp.uint8, device=self.device)
      if previous is not None:
        # Keep buffers alive until all launches using the old binding complete.
        wp.synchronize_stream(wp.get_stream(self.device))
      self.params[sample] = (contents, buffer)
    buffer = self.params[sample][1]
    with self.device.context_guard:
      self._check(
        self.lib.mjwo_launch(self.handle, wp.get_stream(self.device).cuda_stream, buffer.ptr, len(contents), dim[1], dim[0])
      )

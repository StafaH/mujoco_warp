"""GPU population changes using persistent state and one shared physics working batch.

The immutable model catalog supplies specialized physics kernels. Device worklists group
logical worlds by model; each model processes its current worlds in fixed-size chunks.
All model Data layouts alias one CUDA virtual-memory arena, and all captured branches share
one scratch buffer. Persistent integration state and observations retain stable pointers.
"""

import dataclasses
import operator
from collections.abc import Callable
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp._src import types
from mujoco_warp.experimental.heterogeneous import _array_fields
from mujoco_warp.experimental.heterogeneous import allocated_bytes
from mujoco_warp.experimental.pooled_worklist import ModelWorklist

wp.set_module_options({"enable_backward": False})

_STATE_FIELDS = (
  "time",
  "qpos",
  "qvel",
  "act",
  "history",
  "qacc_warmstart",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "eq_active",
  "mocap_pos",
  "mocap_quat",
  "userdata",
)
_SEED_FIELDS = ("xpos", "xquat", "xmat", "ximat", "geom_xpos", "geom_xmat", "overflow")


@wp.kernel
def _reset_persistent_field(
  # In:
  defaults: wp.array[Any],
  models: wp.array[int],
  reset_mask: wp.array[int],
  model_id: int,
  source_width: int,
  state_width: int,
  zero: Any,
  # Out:
  state_out: wp.array[Any],
):
  world, element = wp.tid()
  if reset_mask[world] != 0 and models[world] == model_id:
    value = zero
    if element < source_width:
      value = defaults[element]
    state_out[world * state_width + element] = value


@wp.kernel
def _initialize_working_field(
  # In:
  defaults: wp.array[Any],
  row_width: int,
  # Out:
  values_out: wp.array[Any],
):
  row, element = wp.tid()
  values_out[row * row_width + element] = defaults[element]


@wp.kernel
def _gather_state(
  # In:
  state: wp.array[Any],
  worlds: wp.array[int],
  state_width: int,
  row_width: int,
  # Out:
  values_out: wp.array[Any],
):
  row, element = wp.tid()
  world = worlds[row]
  if world >= 0:
    values_out[row * row_width + element] = state[world * state_width + element]


@wp.kernel
def _scatter_state(
  # In:
  values: wp.array[Any],
  worlds: wp.array[int],
  state_width: int,
  row_width: int,
  # Out:
  state_out: wp.array[Any],
):
  row, element = wp.tid()
  world = worlds[row]
  if world >= 0:
    state_out[world * state_width + element] = values[row * row_width + element]


@wp.kernel
def _begin_model(
  # In:
  offsets: wp.array[int],
  model_id: int,
  # Out:
  cursor_out: wp.array[int],
  limit_out: wp.array[int],
  condition_out: wp.array[int],
):
  cursor_out[0] = offsets[model_id]
  limit_out[0] = offsets[model_id + 1]
  condition_out[0] = int(cursor_out[0] < limit_out[0])


@wp.kernel
def _working_worlds(
  # In:
  worklist: wp.array[int],
  cursor: wp.array[int],
  limit: wp.array[int],
  # Out:
  worlds_out: wp.array[int],
):
  row = wp.tid()
  index = cursor[0] + row
  world = int(-1)
  if index < limit[0]:
    world = worklist[index]
  worlds_out[row] = world


@wp.kernel
def _advance_chunk(
  # In:
  batch_size: int,
  limit: wp.array[int],
  # Out:
  cursor_out: wp.array[int],
  condition_out: wp.array[int],
):
  cursor_out[0] += batch_size
  condition_out[0] = int(cursor_out[0] < limit[0])


@wp.kernel
def _record_diagnostics(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  overflow_in: wp.array[int],
  # In:
  worlds: wp.array[int],
  # Out:
  flags_out: wp.array[int],
  nonfinite_out: wp.array[int],
):
  row = wp.tid()
  world = worlds[row]
  if world >= 0:
    flags_out[world] = flags_out[world] | overflow_in[row]
    invalid = int(0)
    for column in range(qpos_in.shape[1]):
      if not wp.isfinite(qpos_in[row, column]):
        invalid = 1
    for column in range(qvel_in.shape[1]):
      if not wp.isfinite(qvel_in[row, column]):
        invalid = 1
    nonfinite_out[world] = nonfinite_out[world] | invalid


class _SharedDataArena:
  """Startup allocator with overlapping per-model layouts in one VMM reservation."""

  def __init__(self, reserve, device):
    self.storage = wp.VirtualMemory(reserve, device=device)
    self.memory_kind = wp.MemoryKind.CUDA_DEVICE
    self.cursor = 0
    self.highwater_bytes = 0

  def allocate(self, size_in_bytes):
    offset = (self.cursor + 255) // 256 * 256
    end = offset + size_in_bytes
    if end > self.storage.reserved_size:
      raise MemoryError("Shared physics arena exhausted; increase virtual_reserve_bytes")
    if end > self.storage.committed_size:
      self.storage.commit(end)
    self.cursor = end
    self.highwater_bytes = max(self.highwater_bytes, end)
    return self.storage.ptr + offset

  def deallocate(self, ptr, size_in_bytes):
    # Layout aliases are retained until all model views of this arena are released.
    pass


class PooledHeterogeneousWorlds:
  """Process any model mixture entirely on GPU using one reusable physics batch.

  Persistent fields have a logical-world leading dimension and a trailing dimension equal
  to the largest model field width. Only each world's current model prefix is meaningful.
  Integration state is much smaller than collision and solver workspace; those larger Data
  arrays live only in the shared working batch. Sleeping and callbacks are unsupported.
  """

  def __init__(
    self,
    models,
    world_model_ids,
    *,
    batch_size=4096,
    device=None,
    data_kwargs: Mapping[int, Mapping[str, int]] | None = None,
    scratch_bytes=256 << 20,
    observation_fields=("sensordata",),
    virtual_reserve_bytes=4 << 30,
  ):
    """Prepare immutable models, persistent state, shared Data, and capture scratch.

    All virtual-memory pages used by this prototype are committed during startup. Graph
    replay allocates no memory and changes populations without host reset or recapture.
    batch_size bounds working memory; the last chunk of each model may have vacant rows.
    scratch_bytes is a startup-resident capture budget; capture raises if it is exhausted.
    A smaller batch_size and scratch budget trade lower memory for more chunk processing.
    """
    ids = np.asarray(world_model_ids)
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
      raise ValueError("world_model_ids must be a nonempty integer sequence")
    if not len(models) or np.any(ids < 0) or np.any(ids >= len(models)):
      raise ValueError("world_model_ids contains an invalid model index")
    self.batch_size = operator.index(batch_size)
    if self.batch_size < 1:
      raise ValueError("batch_size must be positive")
    self.device = wp.get_device(device)
    if not self.device.is_cuda or not hasattr(wp, "ScopedCaptureScratch") or not hasattr(wp, "VirtualMemory"):
      raise RuntimeError("Pooled physics requires CUDA and the modified Warp capture-scratch/VMM APIs")
    scratch_bytes = operator.index(scratch_bytes)
    if scratch_bytes < 1:
      raise ValueError("scratch_bytes must be positive")
    for model in models:
      if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
        raise ValueError("Pooled physics does not support sleeping; cached asleep state requires additional persistence")
    observations = tuple(dict.fromkeys(observation_fields))
    fields = tuple(dict.fromkeys((*_STATE_FIELDS, *observations)))
    specs = {field.name: field.type for field in dataclasses.fields(types.Data)}
    for name in observations:
      if name not in specs or getattr(specs[name], "shape", (None,))[0] != "nworld":
        raise ValueError(f"Observation {name!r} must be a world-indexed Data array")
    self.nworld = int(ids.size)
    self.graph = None
    self.steps_per_launch = 1
    self._before_reset = None
    self._before_step = None
    self._fields = fields
    self._observations = observations
    self._groups = []
    self._reset_jobs = []
    self._arena = _SharedDataArena(virtual_reserve_bytes, self.device)
    data_kwargs = data_kwargs or {}
    with wp.ScopedDevice(self.device):
      for model_id, host_model in enumerate(models):
        kwargs = dict(data_kwargs.get(model_id, {}))
        if "nworld" in kwargs:
          raise ValueError("data_kwargs must not override nworld")
        model = io.put_model(host_model)
        if not model.opt.run_collision_detection:
          raise ValueError("Pooled physics requires run_collision_detection=True")
        if model.callback.control is not None:
          raise ValueError("Pooled physics requires global policy callbacks rather than Model control callbacks")
        self._arena.cursor = 0
        with wp.ScopedAllocator(self.device, self._arena):
          data = io.make_data(host_model, nworld=self.batch_size, **kwargs)
        defaults = io.make_data(host_model, nworld=1, **kwargs)
        io.reset_data(model, defaults)
        arrays = dict((name, array) for name, _, array in _array_fields(data))
        default_arrays = dict((name, array) for name, _, array in _array_fields(defaults))
        self._groups.append(SimpleNamespace(model=model, data=data, arrays=arrays, defaults=default_arrays))

      self.world_model_ids = wp.array(ids.astype(np.int32), dtype=int)
      self.worklist = ModelWorklist(self.world_model_ids, len(models), device=self.device)
      self.reset_model_ids = wp.full(self.nworld, -1, dtype=int)
      self.device_reset_status = self.worklist.status
      self.device_reset_epoch = self.worklist.epoch
      self._reset_requested = self.worklist.transaction_status[1:2]
      self._working_ids = wp.full(self.batch_size, -1, dtype=int)
      self._cursor = wp.zeros(1, dtype=int)
      self._limit = wp.zeros(1, dtype=int)
      self._condition = wp.zeros(1, dtype=int)
      self._scratch_buffer = wp.empty(scratch_bytes, dtype=wp.uint8)
      self._scratch = wp.ScopedCaptureScratch(self._scratch_buffer)
      self._scratch_peak_bytes = 0
      self.physics_overflow = wp.zeros(self.nworld, dtype=int)
      self.physics_nonfinite = wp.zeros(self.nworld, dtype=int)
      self.state = SimpleNamespace()
      self._widths = {}
      for name in fields:
        samples = [group.arrays[name] for group in self._groups]
        width = max(array.size // self.batch_size for array in samples)
        self._widths[name] = width
        shape = (self.nworld,) if samples[0].ndim == 1 else (self.nworld, width)
        setattr(self.state, name, wp.zeros(shape, dtype=samples[0].dtype))
      for group in self._groups:
        group.initialize = []
        group.gather = []
        group.scatter = []
        for name in dict.fromkeys((*fields, *_SEED_FIELDS)):
          array = group.arrays[name]
          row_width = array.size // self.batch_size
          if row_width:
            group.initialize.append((group.defaults[name].flatten(), row_width, array.flatten()))
        for name in _STATE_FIELDS:
          array = group.arrays[name]
          row_width = array.size // self.batch_size
          if row_width:
            group.gather.append((getattr(self.state, name).flatten(), self._widths[name], row_width, array.flatten()))
        for name in fields:
          array = group.arrays[name]
          row_width = array.size // self.batch_size
          if row_width:
            group.scatter.append((array.flatten(), self._widths[name], row_width, getattr(self.state, name).flatten()))
        reset_jobs = []
        for name in fields:
          width = self._widths[name]
          if width:
            array = getattr(self.state, name)
            defaults = group.defaults[name]
            reset_jobs.append((defaults.flatten(), defaults.size, width, array.dtype(), array.flatten()))
        self._reset_jobs.append(reset_jobs)
        # Retain only template arrays actually copied into persistent/working state.
        group.defaults = {name: group.defaults[name] for name in dict.fromkeys((*fields, *_SEED_FIELDS))}
      initialize_all = wp.ones(self.nworld, dtype=int)
      self._reset_persistent(initialize_all)

  def _reset_persistent(self, reset_mask):
    for model_id, jobs in enumerate(self._reset_jobs):
      for defaults, source_width, width, zero, destination in jobs:
        wp.launch(
          _reset_persistent_field,
          dim=(self.nworld, width),
          inputs=[defaults, self.world_model_ids, reset_mask, model_id, source_width, width, zero],
          outputs=[destination],
          device=self.device,
        )

  def _process_chunk(self, group):
    self._scratch.rewind()
    wp.launch(
      _working_worlds,
      dim=self.batch_size,
      inputs=[self.worklist.model_worlds, self._cursor, self._limit],
      outputs=[self._working_ids],
    )
    for defaults, width, destination in group.initialize:
      wp.launch(_initialize_working_field, dim=(self.batch_size, width), inputs=[defaults, width], outputs=[destination])
    for state, state_width, row_width, destination in group.gather:
      wp.launch(
        _gather_state,
        dim=(self.batch_size, row_width),
        inputs=[state, self._working_ids, state_width, row_width],
        outputs=[destination],
      )
    if not group.model.is_sparse:
      # Dense solver tiles read padded columns and inactive rows. Ordinary Data keeps
      # those initially zero; another model's aliased layout can overwrite them.
      group.data.efc.J.zero_()
      group.data.efc.state.zero_()
    forward.step(group.model, group.data)
    self._scratch_peak_bytes = max(self._scratch_peak_bytes, self._scratch.used_bytes)
    wp.launch(
      _record_diagnostics,
      dim=self.batch_size,
      inputs=[group.data.qpos, group.data.qvel, group.data.overflow, self._working_ids],
      outputs=[self.physics_overflow, self.physics_nonfinite],
    )
    for source, state_width, row_width, state in group.scatter:
      wp.launch(
        _scatter_state,
        dim=(self.batch_size, row_width),
        inputs=[source, self._working_ids, state_width, row_width],
        outputs=[state],
      )
    wp.launch(
      _advance_chunk,
      dim=1,
      inputs=[self.batch_size, self._limit],
      outputs=[self._cursor, self._condition],
    )

  def capture(self, *, steps_per_launch=1, before_reset: Callable | None = None, before_step: Callable | None = None):
    """Capture device requests, persistent policy controls, and all model chunk loops.

    before_reset(self) can read persistent observations and produce reset_model_ids.
    before_step(self) writes persistent controls after valid requests reset integration
    state. Both callbacks enqueue device work; host state mutation/readback is unsupported.
    Callback temporaries must finish before physics starts: each physics branch reuses
    capture scratch. Allocate arrays retained across steps before calling capture.
    """
    steps_per_launch = operator.index(steps_per_launch)
    if steps_per_launch < 1:
      raise ValueError("steps_per_launch must be positive")
    with wp.ScopedDevice(self.device):
      with self._scratch:
        with wp.ScopedCapture(device=self.device) as capture:
          for _ in range(steps_per_launch):
            if before_reset is not None:
              before_reset(self)
            self.worklist.plan(self.reset_model_ids)
            wp.capture_if(self._reset_requested, lambda: self._reset_persistent(self.worklist.reset_mask))
            if before_step is not None:
              before_step(self)
            for model_id, group in enumerate(self._groups):
              wp.launch(
                _begin_model,
                dim=1,
                inputs=[self.worklist.model_offsets, model_id],
                outputs=[self._cursor, self._limit, self._condition],
              )
              wp.capture_while(self._condition, lambda group=group: self._process_chunk(group))
    self.graph = capture.graph
    self.steps_per_launch = steps_per_launch
    self._before_reset = before_reset
    self._before_step = before_step
    return self.graph

  def step(self):
    """Replay the startup graph without host reset, allocation, readback, or recapture."""
    if self.graph is None:
      raise RuntimeError("Capture pooled physics before entering the execution loop")
    wp.capture_launch(self.graph)

  def memory_report(self):
    """Report owned array storage, shared committed Data, and reserved capture scratch."""
    if self.device.is_capturing:
      raise RuntimeError("memory_report is a host diagnostic outside graph capture")
    model_bytes = allocated_bytes([group.model for group in self._groups])
    state_bytes = allocated_bytes(vars(self.state))
    defaults_bytes = allocated_bytes([group.defaults for group in self._groups])
    worklist_bytes = self.worklist.workspace_bytes + allocated_bytes(
      [
        self.world_model_ids,
        self.reset_model_ids,
        self._working_ids,
        self._cursor,
        self._limit,
        self._condition,
        self.physics_overflow,
        self.physics_nonfinite,
      ]
    )
    return {
      "nworld": self.nworld,
      "batch_size": self.batch_size,
      "model_bytes": model_bytes,
      "persistent_state_bytes": state_bytes,
      "defaults_bytes": defaults_bytes,
      "worklist_bytes": worklist_bytes,
      "shared_data_bytes": self._arena.storage.committed_size,
      "shared_data_highwater_bytes": self._arena.highwater_bytes,
      "virtual_reserved_bytes": self._arena.storage.reserved_size,
      "capture_scratch_bytes": self._scratch_buffer.capacity,
      "capture_scratch_used_bytes": self._scratch_peak_bytes,
      "total_bytes": model_bytes
      + state_bytes
      + defaults_bytes
      + worklist_bytes
      + self._arena.storage.committed_size
      + self._scratch_buffer.capacity,
    }

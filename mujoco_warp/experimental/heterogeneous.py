"""Compact model buckets for worlds with unrelated MuJoCo topologies.

Each distinct supplied model gets its own compact batch. One CUDA graph runs all batches,
so a humanoid world never allocates the state or solver workspace of a robot in another
batch. Logical world IDs map to rows through explicit device arrays. The physics kernels
retain their existing homogeneous layout and model specialization.

Logical worlds can switch models using device reset requests inside a captured graph.
Physical capacities are provisioned at startup; overflow rejects the GPU transaction without
a host fallback. Vacant slots still run physics. For arbitrary population mixes that share
workspace across models, use PooledHeterogeneousWorlds. CUDA virtual memory is optional here.
"""

import dataclasses
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp._src import types

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_device_slot_rows(
  # In:
  defaults: wp.array[Any],
  reset_mask: wp.array[int],
  transaction_status: wp.array[int],
  slot_offset: int,
  row_size: int,
  # Out:
  values_out: wp.array[Any],
):
  row, element = wp.tid()
  if transaction_status[0] == 0 and reset_mask[slot_offset + row] != 0:
    values_out[row * row_size + element] = defaults[element]


def _array_fields(value, path=""):
  for field in dataclasses.fields(value):
    item = getattr(value, field.name)
    name = f"{path}.{field.name}" if path else field.name
    if isinstance(item, wp.array):
      yield name, field.type, item
    elif dataclasses.is_dataclass(item):
      yield from _array_fields(item, name)


def _batched_model(model):
  return any(getattr(spec, "shape", (None,))[0] == "*" and array.shape[0] > 1 for _, spec, array in _array_fields(model))


def allocated_bytes(value: Any) -> int:
  """Count array allocation bytes, deduplicating aliases within a model or Data object.

  This reports array storage, excluding CUDA graphs, compiled kernels, allocator pools,
  and other runtime overhead. Use HeterogeneousWorlds.memory_report for virtual storage.
  """
  allocations = {}
  visited = set()

  def visit(obj):
    if id(obj) in visited:
      return
    visited.add(id(obj))
    if isinstance(obj, wp.array):
      if obj.ptr:
        key = (str(obj.device), obj.ptr)
        allocations[key] = max(allocations.get(key, 0), obj.capacity)
    elif dataclasses.is_dataclass(obj):
      for field in dataclasses.fields(obj):
        visit(getattr(obj, field.name))
    elif isinstance(obj, Mapping):
      for item in obj.values():
        visit(item)
    elif isinstance(obj, (tuple, list)):
      for item in obj:
        visit(item)

  visit(value)
  return sum(allocations.values())


class _VirtualArena:
  """Monotonic CUDA VMM allocator; space is reclaimed when the whole arena is released."""

  def __init__(self, size_in_bytes, device):
    self.storage = wp.VirtualMemory(size_in_bytes, device=device)
    self.memory_kind = wp.MemoryKind.CUDA_DEVICE
    self.highwater_bytes = 0
    self.requested_bytes = 0

  def allocate(self, size_in_bytes):
    if self.storage.device.is_capturing:
      raise RuntimeError("Cannot allocate from a virtual arena during graph capture")
    offset = (self.highwater_bytes + 255) // 256 * 256
    end = offset + size_in_bytes
    if end > self.storage.reserved_size:
      raise MemoryError("Virtual Data arena exhausted; increase virtual_reserve_bytes")
    if end > self.storage.committed_size:
      self.storage.commit(end)
    self.highwater_bytes = end
    self.requested_bytes += size_in_bytes
    return self.storage.ptr + offset

  def deallocate(self, ptr, size_in_bytes):
    # Individual allocations can still be referenced by graphs, tensors, and array views.
    # Their memory is deliberately retained until all arena owners have been released.
    pass


@dataclasses.dataclass(frozen=True)
class WorldGroup:
  """One model's physical slots; world_ids contains -1 for unassigned slots."""

  model_id: int
  model: types.Model
  data: types.Data
  world_ids: wp.array
  virtual_memory: Mapping[str, Any]
  arena: Any = None


class HeterogeneousWorlds:
  """Run arbitrary MuJoCo models in compact batches under a single CUDA graph.

  Models are grouped by supplied model ID, not by a guessed topology fingerprint. Models
  with equal dimensions can have different dynamics and must not be silently merged.
  Fields returned by arrays() and world_array() are zero-copy Warp views; wp.to_torch()
  exports them to learning frameworks without padding unlike models to a common shape.
  """

  def __init__(
    self,
    models: Sequence[mujoco.MjModel],
    world_model_ids: Sequence[int],
    *,
    device=None,
    data_kwargs: Mapping[int, Mapping[str, int]] | None = None,
    memory: str = "default",
    virtual_reserve_bytes: int | Mapping[int, int] = 4 << 30,
    capacity_per_model: Mapping[int, int] | None = None,
    contact_storage: Any = None,
  ):
    """Build compact batches, initializing each row from its model's default state.

    Args:
      models: Host models indexed by world_model_ids. Unused models allocate no storage.
      world_model_ids: Model index for each logical world, in caller order.
      device: Warp device; defaults to Warp's current device.
      data_kwargs: Per-model make_data options, such as nconmax, njmax, or nvmax.
      memory: "default", or "virtual" for one CUDA VMM arena per batch's Data arrays.
      virtual_reserve_bytes: Virtual arena size, shared or specified per model ID. Pages
        are committed on demand during construction; individual frees do not reclaim space.
      capacity_per_model: Physical world slots per model. Defaults to exact initial counts.
        Extra slots allow switching models without graph recapture, but still run physics.
      contact_storage: Experimental shared native contact owner. Each complete model step
        reuses its columns; consume contacts before the next group. Persistent contact
        inspection and manually interleaved step1/step2 calls are incompatible.
    """
    ids = np.asarray(world_model_ids)
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
      raise ValueError("world_model_ids must be a nonempty one-dimensional integer sequence")
    if not len(models) or np.any(ids < 0) or np.any(ids >= len(models)):
      raise ValueError("world_model_ids contains an invalid model index")
    if memory not in ("default", "virtual"):
      raise ValueError("memory must be 'default' or 'virtual'")
    self.device = wp.get_device(device)
    if contact_storage is not None and contact_storage.device != self.device:
      raise ValueError("contact_storage must be on the worlds device")
    self.contact_storage = contact_storage
    if memory == "virtual" and (not self.device.is_cuda or not hasattr(wp, "VirtualMemory")):
      raise RuntimeError("virtual storage requires a CUDA device and Warp's VirtualMemory API")
    self.nworld = int(ids.size)
    self.graph = None
    self.steps_per_launch = 1
    self._before_step = None
    self._before_reset = None
    self._after_group = None
    self.device_reset = None
    self.device_reset_status = None
    self.device_reset_epoch = None
    self.reset_model_ids = None
    self._device_reset_jobs = None
    self._models = tuple(models)
    self._data_kwargs = {key: dict(value) for key, value in (data_kwargs or {}).items()}
    self._memory = memory
    self._virtual_reserve_bytes = virtual_reserve_bytes
    self._host_model_ids = ids.astype(np.int32)
    self._host_group = np.empty(self.nworld, dtype=np.int32)
    self._host_row = np.empty(self.nworld, dtype=np.int32)
    self._host_slots = []
    groups = []
    capacities = dict(capacity_per_model or {})
    for model_id, capacity in capacities.items():
      if not isinstance(model_id, (int, np.integer)) or model_id < 0 or model_id >= len(models):
        raise ValueError("capacity_per_model contains an invalid model index")
      if not isinstance(capacity, (int, np.integer)) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("model capacity must be a nonnegative integer")
    for model_id in np.unique(ids):
      count = int(np.count_nonzero(ids == model_id))
      capacities.setdefault(int(model_id), count)
      if capacities[model_id] < count:
        raise ValueError("model capacity cannot be smaller than its initial world count")
    with wp.ScopedDevice(self.device):
      for model_id in sorted(capacities):
        if not capacities[model_id]:
          continue
        world_ids = np.flatnonzero(ids == model_id).astype(np.int32)
        slots = np.full(capacities[model_id], -1, dtype=np.int32)
        slots[: len(world_ids)] = world_ids
        self._host_group[world_ids] = len(groups)
        self._host_row[world_ids] = np.arange(len(world_ids), dtype=np.int32)
        groups.append(self._make_group(model_id, slots))
        self._host_slots.append(slots)
      self.world_model_ids = wp.array(ids.astype(np.int32), dtype=int)
      self.world_group = wp.array(self._host_group, dtype=int)
      self.world_row = wp.array(self._host_row, dtype=int)
    self.groups = tuple(groups)

  def _make_group(self, model_id, slots):
    kwargs = dict(self._data_kwargs.get(model_id, {}))
    if "nworld" in kwargs:
      raise ValueError("data_kwargs must not override nworld")
    model = io.put_model(self._models[model_id])
    if self.contact_storage is not None:
      self.contact_storage.validate_model(model)
      if "contact_storage" in kwargs:
        raise ValueError("Supply contact_storage to HeterogeneousWorlds, not data_kwargs")
      kwargs["contact_storage"] = self.contact_storage
    virtual_memory = {}
    arena = None
    if self._memory == "virtual":
      reserve = self._virtual_reserve_bytes
      reserve = reserve[model_id] if isinstance(reserve, Mapping) else reserve
      arena = _VirtualArena(reserve, self.device)
      with wp.ScopedAllocator(self.device, arena):
        data = io.make_data(self._models[model_id], nworld=len(slots), **kwargs)
      virtual_memory["data"] = arena.storage
    else:
      data = io.make_data(self._models[model_id], nworld=len(slots), **kwargs)
    return WorldGroup(model_id, model, data, wp.array(slots, dtype=int), virtual_memory, arena)

  def locate(self, world_id: int) -> tuple[WorldGroup, int]:
    """Return the batch and physical slot containing a logical world."""
    if self.device_reset is not None:
      raise RuntimeError("locate() uses host mappings; use device world_group/world_row after enabling device resets")
    if not isinstance(world_id, (int, np.integer)) or world_id < 0 or world_id >= self.nworld:
      raise IndexError(f"world ID must be in [0, {self.nworld})")
    return self.groups[self._host_group[world_id]], int(self._host_row[world_id])

  def arrays(self, field: str) -> tuple[tuple[wp.array, wp.array], ...]:
    """Return (world IDs, zero-copy Data array) pairs; IDs of vacant slots are -1."""
    if self.contact_storage is not None and field.startswith("contact."):
      raise RuntimeError("Shared contacts are transient; consume group.data.contact inside after_group")
    result = []
    for group in self.groups:
      value = group.data
      for name in field.split("."):
        value = getattr(value, name)
      if not isinstance(value, wp.array):
        raise ValueError(f"Data field {field!r} is not an array")
      result.append((group.world_ids, value))
    return tuple(result)

  def world_array(self, world_id: int, field: str) -> wp.array:
    """Return one world's zero-copy array view, retaining a leading dimension of one.

    Global contact buffers are shared among a group's worlds and cannot be sliced by row.
    A view binds to a physical slot. Device reset mode disables host lookups because the
    authoritative logical mapping lives on the GPU.
    """
    group, row = self.locate(world_id)
    parent = group.data
    parts = field.split(".")
    for name in parts[:-1]:
      parent = getattr(parent, name)
    spec = parent.__dataclass_fields__[parts[-1]].type
    if not getattr(spec, "shape", ()) or spec.shape[0] != "nworld":
      raise ValueError(f"Data field {field!r} is not indexed by world")
    return getattr(parent, parts[-1])[row : row + 1]

  def enable_device_resets(self) -> wp.array:
    """Prepare fixed-capacity model switching before capture; return device requests.

    Write a destination model ID to reset_model_ids[world], or -1 for no reset. Successful
    transactions consume requests; rejected transactions retain them and set sticky status.
    step() and capture() apply requests immediately before before_step and physics. Their
    before_reset callback can produce requests from device episode state inside the graph.

    All model defaults and scratch are prepared here, outside the execution loop. Physical
    capacities and model configuration are frozen until this runner is discarded. Provision
    slots for every model that may be requested; unused models need capacity_per_model entries.
    Host locate() and world_array() are disabled afterward. Export physical
    group arrays and route logical worlds with the device world_group/world_row mappings.
    """
    if self.device.is_capturing or self.graph is not None:
      raise RuntimeError("Enable device resets before graph capture")
    if self.device_reset is not None:
      return self.reset_model_ids
    for group in self.groups:
      if _batched_model(group.model):
        raise ValueError("Device resets require shared model parameters")
      if not group.model.opt.run_collision_detection:
        raise ValueError("Device resets require run_collision_detection=True")
    from mujoco_warp.experimental.device_reset import DeviceResetPlanner

    with wp.ScopedDevice(self.device):
      offsets = np.concatenate(([0], np.cumsum([group.data.nworld for group in self.groups]))).astype(np.int32)
      model_groups = np.full(len(self._models), -1, dtype=np.int32)
      model_groups[[group.model_id for group in self.groups]] = np.arange(len(self.groups), dtype=np.int32)
      slots = wp.array(np.concatenate(self._host_slots), dtype=int)
      planner = DeviceResetPlanner(
        wp.array(offsets, dtype=int),
        wp.array([group.model_id for group in self.groups], dtype=int),
        wp.array(model_groups, dtype=int),
        slots,
        self.world_group,
        self.world_row,
        self.world_model_ids,
        device=self.device,
      )
      jobs = []
      for group_id, group in enumerate(self.groups):
        defaults = io.make_data(self._models[group.model_id], nworld=1, **self._data_kwargs.get(group.model_id, {}))
        io.reset_data(group.model, defaults)
        fields = dict((name, array) for name, _, array in _array_fields(defaults))
        reset_jobs = []
        for name, spec, array in _array_fields(group.data):
          if getattr(spec, "shape", (None,))[0] != "nworld" or not array.size:
            continue
          row_size = array.size // group.data.nworld
          if fields[name].size != row_size:
            raise ValueError(f"Data field {name} does not match the reset model layout")
          reset_jobs.append((fields[name].flatten(), row_size, array.flatten()))
        jobs.append((int(offsets[group_id]), group.data.nworld, planner.group_reset_count[group_id : group_id + 1], reset_jobs))
      requests = wp.full(self.nworld, -1, dtype=int)
      self.groups = tuple(
        dataclasses.replace(group, world_ids=slots[int(offsets[group_id]) : int(offsets[group_id + 1])])
        for group_id, group in enumerate(self.groups)
      )
    self._device_reset_jobs = jobs
    self.reset_model_ids = requests
    self.device_reset = planner
    self.device_reset_status = planner.status
    self.device_reset_epoch = planner.epoch
    return requests

  def reset_worlds_device(self, requests: wp.array | None = None):
    """Enqueue a GPU reset transaction without allocation, synchronization, or host reads.

    This maintenance phase must be followed immediately by physics. Contact observations
    are valid before before_reset or after physics; before_step must not consume old contacts.
    Unchanged worlds retain their slots/state. Capacity failure is atomic and reported on
    device_reset_status, allowing device-side retry or backpressure without breaking capture.
    """
    if self.device_reset is None:
      raise RuntimeError("Call enable_device_resets() before enqueuing device resets")
    requests = self.reset_model_ids if requests is None else requests
    with wp.ScopedDevice(self.device):
      self.device_reset.plan(requests)
      for offset, capacity, condition, jobs in self._device_reset_jobs:

        def reset_group(offset=offset, capacity=capacity, jobs=jobs):
          for defaults, row_size, destination in jobs:
            wp.launch(
              _reset_device_slot_rows,
              dim=(capacity, row_size),
              inputs=[defaults, self.device_reset.reset_mask, self.device_reset.transaction_status, offset, row_size],
              outputs=[destination],
            )

        if self.device.is_cuda and self.device.is_capturing:
          wp.capture_if(condition, reset_group)
        else:
          # capture_if outside capture reads its condition on the host; guarded launches
          # keep eager execution entirely asynchronous as well.
          reset_group()

  def inspect_layout(self) -> dict:
    """Read a diagnostic mapping snapshot outside the execution loop and graph capture."""
    if self.device.is_capturing:
      raise RuntimeError("Cannot inspect the world layout during graph capture")
    wp.synchronize_device(self.device)
    return {
      "world_model_ids": self.world_model_ids.numpy(),
      "world_group": self.world_group.numpy(),
      "world_row": self.world_row.numpy(),
      "group_world_ids": tuple(group.world_ids.numpy() for group in self.groups),
    }

  def capture(
    self,
    *,
    steps_per_launch: int = 1,
    before_step: Callable | None = None,
    before_reset: Callable | None = None,
    after_group: Callable | None = None,
  ):
    """Capture all batches and an optional device-work callback in one CUDA graph.

    Each timestep enqueues before_reset(self), device reset maintenance when enabled,
    before_step(self), and physics. Callbacks enqueue device work; they must not mutate host
    state. before_reset may read previous contacts with independent storage, which are stale
    until physics after reset. Shared contacts are only valid in after_group.
    after_group(self, group) consumes current contacts immediately after a complete model
    step, before shared contact storage is overwritten by the next group.
    Capture itself does not advance simulation. Physical capacities and model specialization
    remain fixed throughout replay.
    """
    if not self.device.is_cuda:
      raise RuntimeError("CUDA graph capture requires a CUDA device")
    if not isinstance(steps_per_launch, int) or steps_per_launch < 1:
      raise ValueError("steps_per_launch must be a positive integer")
    if self.contact_storage is not None:
      for group in self.groups:
        self.contact_storage.validate_model(group.model)
    with wp.ScopedDevice(self.device):
      with wp.ScopedCapture(device=self.device) as capture:
        for _ in range(steps_per_launch):
          if before_reset is not None:
            before_reset(self)
          if self.device_reset is not None:
            self.reset_worlds_device()
          if before_step is not None:
            before_step(self)
          for group in self.groups:
            forward.step(group.model, group.data)
            if self.contact_storage is not None:
              self.contact_storage.observe(group.data)
            if after_group is not None:
              after_group(self, group)
    self.graph = capture.graph
    self.steps_per_launch = steps_per_launch
    self._before_step = before_step
    self._before_reset = before_reset
    self._after_group = after_group
    return self.graph

  def step(self):
    """Advance one timestep eagerly, or steps_per_launch timesteps with a captured graph."""
    with wp.ScopedDevice(self.device):
      if self.graph is not None:
        wp.capture_launch(self.graph)
      else:
        if self._before_reset is not None:
          self._before_reset(self)
        if self.device_reset is not None:
          self.reset_worlds_device()
        if self._before_step is not None:
          self._before_step(self)
        for group in self.groups:
          if self.contact_storage is not None:
            self.contact_storage.validate_model(group.model)
          forward.step(group.model, group.data)
          if self.contact_storage is not None:
            self.contact_storage.observe(group.data)
          if self._after_group is not None:
            self._after_group(self, group)

  def memory_report(self) -> dict:
    """Report allocated array bytes and committed/reserved virtual bytes per batch."""
    if self.device.is_capturing:
      raise RuntimeError("memory_report() is a host diagnostic and cannot run during capture")
    slots = self._host_slots if self.device_reset is None else [group.world_ids.numpy() for group in self.groups]
    groups = []
    for group_id, group in enumerate(self.groups):
      data_arrays = [array for name, _, array in _array_fields(group.data) if not name.startswith("contact.")]
      data_bytes = 0 if group.arena else allocated_bytes(data_arrays if self.contact_storage is not None else group.data)
      committed = sum(storage.committed_size for storage in group.virtual_memory.values())
      reserved = sum(storage.reserved_size for storage in group.virtual_memory.values())
      groups.append(
        {
          "model_id": group.model_id,
          "nworld": group.data.nworld,
          "capacity": group.data.nworld,
          "active_worlds": int(np.count_nonzero(slots[group_id] >= 0)),
          "inactive_slots": int(np.count_nonzero(slots[group_id] < 0)),
          "model_bytes": allocated_bytes(group.model),
          "data_bytes": data_bytes + committed,
          "virtual_committed_bytes": committed,
          "virtual_reserved_bytes": reserved,
          "arena_requested_bytes": group.arena.requested_bytes if group.arena else 0,
          "arena_highwater_bytes": group.arena.highwater_bytes if group.arena else 0,
        }
      )
    slot_arrays = [group.world_ids for group in self.groups]
    device_reset_bytes = 0
    if self.device_reset is not None:
      # The group slot arrays are slices of one packed allocation; count its owner once.
      slot_arrays = [self.device_reset.slot_world_ids]
      device_reset_bytes = self.device_reset.workspace_bytes + allocated_bytes(
        [
          self.device_reset.group_offsets,
          self.device_reset.group_model_ids,
          self.device_reset.model_to_group,
          self.reset_model_ids,
          *[defaults for _, _, _, jobs in self._device_reset_jobs for defaults, _, _ in jobs],
        ]
      )
    mapping_bytes = allocated_bytes([self.world_model_ids, self.world_group, self.world_row, *slot_arrays])
    shared_contact_bytes = self.contact_storage.allocated_bytes if self.contact_storage is not None else 0
    return {
      "groups": groups,
      "mapping_bytes": mapping_bytes,
      "device_reset_bytes": device_reset_bytes,
      "active_worlds": self.nworld,
      "physical_slots": sum(group.data.nworld for group in self.groups),
      "inactive_slots": sum(group["inactive_slots"] for group in groups),
      "shared_contact_bytes": shared_contact_bytes,
      "total_bytes": sum(g["model_bytes"] + g["data_bytes"] for g in groups)
      + mapping_bytes
      + device_reset_bytes
      + shared_contact_bytes,
    }

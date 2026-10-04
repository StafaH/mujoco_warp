"""Shared native contact columns for sequential heterogeneous physics batches.

This specializes DynamicTable's stable-address storage primitive (CUDA virtual
memory), rather than its ECS append, deletion, or compaction machinery. Native
collision appends with ``Data.nacon`` and the solver consumes the same columns.
Capacity is committed at construction; no growth or contact copies occur in a graph.
"""

import dataclasses
import operator

import warp as wp

from mujoco_warp._src import types

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _observe_overflow(
  # Data in:
  nacon_in: wp.array[int],
  ncollision_in: wp.array[int],
  # In:
  capacity: int,
  # Out:
  errors_out: wp.array[int],
):
  if nacon_in[0] > capacity or ncollision_in[0] > capacity:
    wp.atomic_or(errors_out, 0, 1)


class NativeContactTable:
  """Own one transient native Contact arena shared by complete sequential steps.

  Bind contacts during Data construction and consume each group's contacts before
  stepping the next group. Earlier groups' contact arrays then contain stale data.
  Do not interleave ``step1``/``step2`` across groups, call ``reset_data`` against
  stale contacts, or retain contacts as persistent input. Sleeping's two collision
  passes remain inside one complete step and keep their native counter semantics.
  All users must execute in order on one device stream.
  """

  def __init__(self, capacity, max_pyramid=10, include_flex=False, device=None, memory="virtual"):
    """Allocate and initialize fixed-capacity native columns before graph capture."""
    capacity = operator.index(capacity)
    max_pyramid = operator.index(max_pyramid)
    if not 0 < capacity < 2**31:
      raise ValueError("capacity must be between 1 and 2**31 - 1")
    if max_pyramid < 1:
      raise ValueError("max_pyramid must be positive")
    if memory not in ("virtual", "default"):
      raise ValueError("memory must be 'virtual' or 'default'")
    self.device = wp.get_device(device)
    if self.device.is_capturing:
      raise RuntimeError("Native contact storage must be constructed outside graph capture")
    if memory == "virtual" and (not self.device.is_cuda or not hasattr(wp, "VirtualMemory")):
      raise RuntimeError("virtual contact storage requires CUDA and Warp's VirtualMemory API")
    self.capacity = capacity
    self.max_pyramid = max_pyramid
    self.include_flex = bool(include_flex)
    self.memory = memory
    layout = {}
    size = 0
    for field in dataclasses.fields(types.Contact):
      dtype = field.type.dtype
      count = 0 if field.name in ("flex", "elem", "vert") and not self.include_flex else capacity
      shape = (count, max_pyramid) if field.name == "efc_address" else (count,)
      size = (size + 255) // 256 * 256
      layout[field.name] = (size, dtype, shape)
      size += count * wp.types.type_size_in_bytes(dtype) * (max_pyramid if field.name == "efc_address" else 1)
    if memory == "virtual":
      self.storage = wp.VirtualMemory(size, device=self.device)
      self.storage.commit(size)
      self._arena = self.storage.array(size, dtype=wp.uint8)
      self._arena.zero_()
    else:
      self.storage = wp.zeros(size, dtype=wp.uint8, device=self.device)
      self._arena = self.storage
    self.columns = {}
    for name, (offset, dtype, shape) in layout.items():
      view = wp.array(ptr=self._arena.ptr + offset, dtype=dtype, shape=shape, device=self.device, copy=False)
      view._ref = self._arena
      self.columns[name] = view
    self.columns["efc_address"].fill_(-1)
    self.errors = wp.zeros(1, dtype=int, device=self.device)

  @property
  def allocated_bytes(self):
    """Owned physical arena and sticky error bytes, including CUDA page rounding."""
    arena_bytes = self.storage.committed_size if self.memory == "virtual" else self.storage.capacity
    return arena_bytes + self.errors.capacity

  @property
  def bytes(self):
    """Alias for allocated_bytes."""
    return self.allocated_bytes

  def validate_model(self, model):
    """Reject persistent contact input that cannot share transient storage."""
    if not model.opt.run_collision_detection:
      raise ValueError("shared native contacts require run_collision_detection=True")
    if model.nflex > 0 and not self.include_flex:
      raise ValueError("flex models require include_flex=True")

  def make_contact(self, model, naconmax, nmaxpyramid):
    """Return native Contact views with model-specific shapes and shared row strides."""
    naconmax = operator.index(naconmax)
    nmaxpyramid = operator.index(nmaxpyramid)
    if not 0 <= naconmax <= self.capacity:
      raise ValueError("native contact capacity exceeds shared storage")
    if not 1 <= nmaxpyramid <= self.max_pyramid:
      raise ValueError("native pyramid width exceeds shared storage")
    if model.nflex > 0 and not self.include_flex:
      raise ValueError("flex models require include_flex=True")
    views = {}
    for name, column in self.columns.items():
      count = 0 if name in ("flex", "elem", "vert") and model.nflex == 0 else naconmax
      shape = (count, nmaxpyramid) if name == "efc_address" else (count,)
      view = wp.array(
        ptr=column.ptr,
        dtype=column.dtype,
        shape=shape,
        strides=column.strides,
        device=self.device,
        copy=False,
      )
      view._ref = column
      views[name] = view
    return types.Contact(**views)

  def observe(self, data):
    """Record native overflow on the GPU before the next group overwrites contacts."""
    if data.nacon.device != self.device:
      raise ValueError("native contacts must be on the storage device")
    if data.naconmax > self.capacity:
      raise ValueError("native contact capacity exceeds shared storage")
    wp.launch(
      _observe_overflow,
      dim=1,
      inputs=[data.nacon, data.ncollision, data.naconmax],
      outputs=[self.errors],
      device=self.device,
    )

  def clear_errors(self):
    """Clear sticky overflow errors on the device."""
    self.errors.zero_()

  def check_errors(self):
    """Synchronize outside capture and raise if any observed native batch overflowed."""
    if self.errors.numpy()[0]:
      raise RuntimeError("Native contact/pair capacity exhausted; increase capacity and recompute contacts")

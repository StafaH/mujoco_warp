"""Plan fixed-capacity world reassignment entirely on the device."""

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def _count_requests(
  # In:
  requests_in: wp.array[int],
  model_to_group_in: wp.array[int],
  world_group_in: wp.array[int],
  # Out:
  target_group_out: wp.array[int],
  outgoing_out: wp.array[int],
  incoming_out: wp.array[int],
  transaction_out: wp.array[int],
):
  worldid = wp.tid()
  target_group_out[worldid] = -1
  modelid = requests_in[worldid]
  if modelid == -1:
    return
  if modelid < 0 or modelid >= model_to_group_in.shape[0]:
    wp.atomic_or(transaction_out, 0, 1)
    return
  group = model_to_group_in[modelid]
  if group < 0 or group >= incoming_out.shape[0]:
    wp.atomic_or(transaction_out, 0, 1)
    return
  target_group_out[worldid] = group
  wp.atomic_add(outgoing_out, world_group_in[worldid], 1)
  wp.atomic_add(incoming_out, group, 1)
  wp.atomic_add(transaction_out, 1, 1)


@wp.kernel
def _count_occupancy(
  # In:
  slot_world_in: wp.array[int],
  slot_group_in: wp.array[int],
  # Out:
  occupancy_out: wp.array[int],
):
  slot = wp.tid()
  if slot_world_in[slot] >= 0:
    wp.atomic_add(occupancy_out, slot_group_in[slot], 1)


@wp.kernel
def _validate_capacity(
  # In:
  group_offsets_in: wp.array[int],
  occupancy_in: wp.array[int],
  outgoing_in: wp.array[int],
  incoming_in: wp.array[int],
  # Out:
  transaction_out: wp.array[int],
):
  group = wp.tid()
  capacity = group_offsets_in[group + 1] - group_offsets_in[group]
  available = capacity - occupancy_in[group] + outgoing_in[group]
  if incoming_in[group] > available:
    wp.atomic_or(transaction_out, 0, 2)


@wp.kernel
def _release_selected(
  # In:
  transaction_in: wp.array[int],
  target_group_in: wp.array[int],
  group_offsets_in: wp.array[int],
  world_group_in: wp.array[int],
  world_row_in: wp.array[int],
  # Out:
  slot_world_out: wp.array[int],
  reset_mask_out: wp.array[int],
):
  worldid = wp.tid()
  if transaction_in[0] != 0 or target_group_in[worldid] < 0:
    return
  slot = group_offsets_in[world_group_in[worldid]] + world_row_in[worldid]
  slot_world_out[slot] = -1
  reset_mask_out[slot] = 1


@wp.kernel
def _build_free_rows(
  # In:
  transaction_in: wp.array[int],
  slot_world_in: wp.array[int],
  slot_group_in: wp.array[int],
  group_offsets_in: wp.array[int],
  # Out:
  free_cursor_out: wp.array[int],
  free_rows_out: wp.array[int],
):
  slot = wp.tid()
  if transaction_in[0] != 0 or slot_world_in[slot] >= 0:
    return
  group = slot_group_in[slot]
  begin = group_offsets_in[group]
  index = begin + wp.atomic_add(free_cursor_out, group, 1)
  free_rows_out[index] = slot - begin


@wp.kernel
def _assign_selected(
  # In:
  transaction_in: wp.array[int],
  target_group_in: wp.array[int],
  group_offsets_in: wp.array[int],
  group_model_in: wp.array[int],
  free_rows_in: wp.array[int],
  # Out:
  incoming_cursor_out: wp.array[int],
  slot_world_out: wp.array[int],
  world_group_out: wp.array[int],
  world_row_out: wp.array[int],
  world_model_out: wp.array[int],
  requests_out: wp.array[int],
  reset_mask_out: wp.array[int],
  epoch_out: wp.array[wp.uint64],
):
  worldid = wp.tid()
  group = target_group_in[worldid]
  if transaction_in[0] != 0 or group < 0:
    return
  begin = group_offsets_in[group]
  index = begin + wp.atomic_add(incoming_cursor_out, group, 1)
  row = free_rows_in[index]
  slot = begin + row
  slot_world_out[slot] = worldid
  world_group_out[worldid] = group
  world_row_out[worldid] = row
  world_model_out[worldid] = group_model_in[group]
  requests_out[worldid] = -1
  reset_mask_out[slot] = 1
  epoch_out[worldid] += wp.uint64(1)


@wp.kernel
def _count_reset_slots(
  # In:
  reset_mask_in: wp.array[int],
  slot_group_in: wp.array[int],
  # Out:
  group_reset_count_out: wp.array[int],
):
  slot = wp.tid()
  if reset_mask_in[slot] != 0:
    wp.atomic_add(group_reset_count_out, slot_group_in[slot], 1)


@wp.kernel
def _record_status(
  # Out:
  transaction_out: wp.array[int],
  status_out: wp.array[int],
):
  flags = transaction_out[0]
  if flags != 0:
    transaction_out[1] = 0
    wp.atomic_or(status_out, 0, flags)


class DeviceResetPlanner:
  """Reassign logical worlds to preallocated model slots during graph replay.

  Catalog arrays, packed slots, and mapping arrays must be one-dimensional int32
  arrays on the selected device. ``group_offsets`` partitions the packed slots;
  ``group_model_ids`` identifies each group's model. ``model_to_group`` maps each
  catalog model ID to its group, or -1 when the model is not provisioned. Occupied
  slots contain logical world IDs; vacant slots contain -1. World rows are local
  to their group. Construction validates these relationships on the host once.

  ``plan(requests)`` consumes int32 model IDs from device producers. -1 means no
  reset. Same-model requests still reset the world. A transaction commits only
  if every request is valid and all destinations fit after selected source slots
  are released. Failure leaves requests, packed slots, mappings, and epochs
  unchanged. Successful requests become -1. Unselected world rows stay fixed;
  selected worlds may receive a different row. Assignment order is unspecified.

  Reset planning does not initialize simulation state. After planning, callers
  must reset every flagged physical row before executing physics. The mask also
  flags released source rows, keeping vacant slots clear. This API requires
  exclusive stream ordering and forbids concurrent mapping/slot mutation.

  Attributes:
    reset_mask: int32 packed physical-row mask, cleared before each transaction.
    transaction_status: int32[2]: current error bits and committed request count.
    status: int32[1]: sticky error bits, cleared by ``clear_status()``.
    group_reset_count: Number of unique physical reset rows per group.
    epoch: uint64 generation per logical world, incremented on successful reset.

  Error bits are 1 for invalid or unprovisioned model IDs and 2 for capacity
  exhaustion. Rejected requests remain pending and can be replaced by producers.
  All storage and conditional-node views are allocated at construction; planning
  performs no host read, allocation, synchronization, or graph recapture.
  """

  def __init__(
    self,
    group_offsets,
    group_model_ids,
    model_to_group,
    slot_world_ids,
    world_group,
    world_row,
    world_model_ids,
    device=None,
  ):
    """Allocate planner scratch and validate the fixed slot catalog."""
    self.device = wp.get_device(device if device is not None else slot_world_ids.device)
    arrays = (group_offsets, group_model_ids, model_to_group, slot_world_ids, world_group, world_row, world_model_ids)
    if any(
      not isinstance(array, wp.array) or array.device != self.device or array.dtype != wp.int32 or array.ndim != 1
      for array in arrays
    ):
      raise ValueError("Planner inputs must be one-dimensional int32 Warp arrays on the selected device")
    offsets, models, lookup, slots, groups, rows, world_models = [array.numpy() for array in arrays]
    self.nworld = world_group.size
    self.ngroup = group_model_ids.size
    self.nslot = slot_world_ids.size
    if not self.nworld or not self.ngroup or not model_to_group.size:
      raise ValueError("Planner requires nonempty worlds, groups, and model catalog")
    if world_row.shape != world_group.shape or world_model_ids.shape != world_group.shape:
      raise ValueError("Logical world mappings must have equal shape")
    if offsets.shape != (self.ngroup + 1,) or offsets[0] != 0 or offsets[-1] != self.nslot or np.any(np.diff(offsets) < 0):
      raise ValueError("group_offsets must partition the packed slot array")
    if self.nslot >= 2**31:
      raise ValueError("Packed capacity must fit signed int32 indexing")
    if np.any(models < 0) or np.any(models >= lookup.size) or len(np.unique(models)) != self.ngroup:
      raise ValueError("Each group must identify a unique valid catalog model")
    if np.any(lookup < -1) or np.any(lookup >= self.ngroup) or np.any(lookup[models] != np.arange(self.ngroup)):
      raise ValueError("model_to_group must map each provisioned model to its group")
    provisioned = np.flatnonzero(lookup >= 0)
    if np.any(models[lookup[provisioned]] != provisioned):
      raise ValueError("model_to_group contains an inconsistent model alias")
    capacity = np.diff(offsets)
    if np.any(groups < 0) or np.any(groups >= self.ngroup):
      raise ValueError("Logical worlds contain an invalid group")
    if np.any(rows < 0) or np.any(rows >= capacity[groups]) or np.any(world_models != models[groups]):
      raise ValueError("Logical worlds contain an invalid row or model")
    if np.any(slots < -1) or np.any(slots >= self.nworld):
      raise ValueError("Packed slots contain an invalid logical world ID")
    positions = offsets[groups] + rows
    if len(np.unique(positions)) != self.nworld or np.any(slots[positions] != np.arange(self.nworld)):
      raise ValueError("Packed slots and logical mappings disagree")
    if np.count_nonzero(slots >= 0) != self.nworld:
      raise ValueError("Each logical world must occupy exactly one slot")
    self.group_offsets = group_offsets
    self.group_model_ids = group_model_ids
    self.model_to_group = model_to_group
    self.slot_world_ids = slot_world_ids
    self.world_group = world_group
    self.world_row = world_row
    self.world_model_ids = world_model_ids
    with wp.ScopedDevice(self.device):
      self.reset_mask = wp.zeros(self.nslot, dtype=int)
      self.transaction_status = wp.zeros(2, dtype=int)
      self.status = wp.zeros(1, dtype=int)
      self.group_reset_count = wp.zeros(self.ngroup, dtype=int)
      self.epoch = wp.zeros(self.nworld, dtype=wp.uint64)
      self._targets = wp.empty(self.nworld, dtype=int)
      self._outgoing = wp.zeros(self.ngroup, dtype=int)
      self._incoming = wp.zeros(self.ngroup, dtype=int)
      self._occupancy = wp.zeros(self.ngroup, dtype=int)
      self._free_cursor = wp.zeros(self.ngroup, dtype=int)
      self._incoming_cursor = wp.zeros(self.ngroup, dtype=int)
      self._free_rows = wp.empty(self.nslot, dtype=int)
      self._slot_group = wp.array(np.repeat(np.arange(self.ngroup, dtype=np.int32), capacity), dtype=int)
    self._owned_arrays = (
      self.reset_mask,
      self.transaction_status,
      self.status,
      self.group_reset_count,
      self.epoch,
      self._targets,
      self._outgoing,
      self._incoming,
      self._occupancy,
      self._free_cursor,
      self._incoming_cursor,
      self._free_rows,
      self._slot_group,
    )
    self._has_requests = self.transaction_status[1:2]

  @property
  def workspace_bytes(self):
    """Return owned array bytes, excluding bound catalog, packed slots, and mappings."""
    return sum(array.capacity for array in self._owned_arrays)

  def _commit_requested(self):
    wp.launch(
      _count_occupancy,
      dim=self.nslot,
      inputs=[self.slot_world_ids, self._slot_group],
      outputs=[self._occupancy],
      device=self.device,
    )
    wp.launch(
      _validate_capacity,
      dim=self.ngroup,
      inputs=[self.group_offsets, self._occupancy, self._outgoing, self._incoming],
      outputs=[self.transaction_status],
      device=self.device,
    )
    wp.launch(
      _release_selected,
      dim=self.nworld,
      inputs=[self.transaction_status, self._targets, self.group_offsets, self.world_group, self.world_row],
      outputs=[self.slot_world_ids, self.reset_mask],
      device=self.device,
    )
    wp.launch(
      _build_free_rows,
      dim=self.nslot,
      inputs=[self.transaction_status, self.slot_world_ids, self._slot_group, self.group_offsets],
      outputs=[self._free_cursor, self._free_rows],
      device=self.device,
    )
    wp.launch(
      _assign_selected,
      dim=self.nworld,
      inputs=[self.transaction_status, self._targets, self.group_offsets, self.group_model_ids, self._free_rows],
      outputs=[
        self._incoming_cursor,
        self.slot_world_ids,
        self.world_group,
        self.world_row,
        self.world_model_ids,
        self._requests,
        self.reset_mask,
        self.epoch,
      ],
      device=self.device,
    )
    wp.launch(
      _count_reset_slots,
      dim=self.nslot,
      inputs=[self.reset_mask, self._slot_group],
      outputs=[self.group_reset_count],
      device=self.device,
    )

  def plan(self, requests):
    """Record allocation-free reassignment from a device request array.

    CUDA capture records a conditional branch that skips slot maintenance when
    no valid request exists. Producers must run before this operation, and state
    initialization must run afterward in the same stream.
    """
    if not isinstance(requests, wp.array) or requests.device != self.device or requests.dtype != wp.int32:
      raise ValueError("requests must be an int32 Warp array on the planner device")
    if requests.shape != (self.nworld,):
      raise ValueError(f"requests must have shape ({self.nworld},)")
    self._requests = requests
    for array in (
      self.reset_mask,
      self.transaction_status,
      self.group_reset_count,
      self._outgoing,
      self._incoming,
      self._occupancy,
      self._free_cursor,
      self._incoming_cursor,
    ):
      array.zero_()
    wp.launch(
      _count_requests,
      dim=self.nworld,
      inputs=[requests, self.model_to_group, self.world_group],
      outputs=[self._targets, self._outgoing, self._incoming, self.transaction_status],
      device=self.device,
    )
    if self.device.is_cuda and self.device.is_capturing:
      wp.capture_if(self._has_requests, self._commit_requested)
    else:
      self._commit_requested()
    wp.launch(_record_status, dim=1, outputs=[self.transaction_status, self.status], device=self.device)

  def clear_status(self):
    """Clear sticky error bits; recordable inside the captured execution loop."""
    self.status.zero_()

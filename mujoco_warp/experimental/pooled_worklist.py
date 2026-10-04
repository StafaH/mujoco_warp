"""Maintain model-grouped logical worlds without per-model occupancy limits."""

import operator

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def _validate_requests(
  # In:
  num_models: int,
  requests_in: wp.array[int],
  # Out:
  targets_out: wp.array[int],
  transaction_out: wp.array[int],
):
  worldid = wp.tid()
  modelid = requests_in[worldid]
  targets_out[worldid] = -1
  if modelid == -1:
    return
  if modelid < 0 or modelid >= num_models:
    wp.atomic_or(transaction_out, 0, 1)
    return
  targets_out[worldid] = modelid
  wp.atomic_add(transaction_out, 1, 1)


@wp.kernel
def _commit_requests(
  # In:
  targets_in: wp.array[int],
  transaction_in: wp.array[int],
  # Out:
  world_models_out: wp.array[int],
  requests_out: wp.array[int],
  reset_mask_out: wp.array[int],
  epoch_out: wp.array[wp.uint64],
  changed_out: wp.array[int],
):
  worldid = wp.tid()
  modelid = targets_in[worldid]
  if transaction_in[0] != 0 or modelid < 0:
    return
  if world_models_out[worldid] != modelid:
    wp.atomic_or(changed_out, 0, 1)
  world_models_out[worldid] = modelid
  reset_mask_out[worldid] = 1
  epoch_out[worldid] += wp.uint64(1)
  requests_out[worldid] = -1


@wp.kernel
def _model_histogram(
  # In:
  world_models_in: wp.array[int],
  # Out:
  counts_out: wp.array[int],
):
  worldid = wp.tid()
  wp.atomic_add(counts_out, world_models_in[worldid], 1)


@wp.kernel
def _prefix_model_counts(
  # In:
  counts_in: wp.array[int],
  # Out:
  offsets_out: wp.array[int],
):
  total = int(0)
  for modelid in range(counts_in.shape[0]):
    offsets_out[modelid] = total
    total += counts_in[modelid]
  offsets_out[counts_in.shape[0]] = total


@wp.kernel
def _scatter_worlds(
  # In:
  world_models_in: wp.array[int],
  offsets_in: wp.array[int],
  # Out:
  cursors_out: wp.array[int],
  model_worlds_out: wp.array[int],
):
  worldid = wp.tid()
  modelid = world_models_in[worldid]
  index = offsets_in[modelid] + wp.atomic_add(cursors_out, modelid, 1)
  model_worlds_out[index] = worldid


@wp.kernel
def _record_errors(
  # Out:
  transaction_out: wp.array[int],
  status_out: wp.array[int],
):
  if transaction_out[0] != 0:
    transaction_out[1] = 0
    wp.atomic_or(status_out, 0, transaction_out[0])


class ModelWorklist:
  """Group logical worlds by model for fixed-size working-batch chunk loops.

  Every catalog model can serve any number of the ``nworld`` logical worlds.
  ``model_worlds`` stores all logical world IDs grouped by model; model ``m``
  owns ``[model_offsets[m], model_offsets[m+1])``. Relative ordering within a
  model is unspecified. These buffers keep their shapes and pointers fixed.

  Device producers write model IDs to a request array; -1 means no reset.
  ``plan(requests)`` globally validates all requests before changing any model,
  epoch, or request. On success every selected logical world is flagged for
  default-state initialization and its request becomes -1. Same-model resets
  still flag state initialization and increment epochs. Invalid requests remain
  pending, and the entire transaction changes no assignments or epochs.

  During graph replay, worklists rebuild only when a successful reset changes a
  model assignment. Eager calls rebuild unconditionally without CPU readback.
  Catalog prefix sums use one GPU thread over models; world histogram/scatter
  use parallel atomics. All scratch is allocated at construction. No per-model
  physical capacity, CPU readback, allocation, or synchronization is needed in
  ``plan``. State initialization and physics belong to the caller, ordered on
  the same stream immediately after planning.

  Attributes:
    world_model_ids: Bound device int32 model assignment per logical world.
    model_worlds: Model-grouped logical world IDs, length ``nworld``.
    model_offsets: Start/end offsets, length ``num_models + 1``.
    model_counts: Number of logical worlds assigned to each model.
    reset_mask: int32 per-world default-state reset mask for this transaction.
    epoch: uint64 reset generation per logical world.
    transaction_status: int32[2]: current error bits and committed reset count.
    status: int32[1]: sticky error bits. Bit 1 denotes an invalid model ID.
  """

  def __init__(self, world_model_ids, num_models, device=None):
    """Validate initial assignments and allocate worklists and transaction scratch."""
    if not isinstance(world_model_ids, wp.array):
      raise ValueError("world_model_ids must be a Warp array on the selected device")
    self.device = wp.get_device(device if device is not None else world_model_ids.device)
    num_models = operator.index(num_models)
    if not 0 < num_models < 2**31 - 1:
      raise ValueError("num_models must be positive and support an offset sentinel")
    if world_model_ids.device != self.device:
      raise ValueError("world_model_ids must be a Warp array on the selected device")
    if world_model_ids.dtype != wp.int32 or world_model_ids.ndim != 1 or not world_model_ids.size:
      raise ValueError("world_model_ids must be a nonempty one-dimensional int32 array")
    initial = world_model_ids.numpy()
    if np.any(initial < 0) or np.any(initial >= num_models):
      raise ValueError("Initial world model IDs are outside the catalog")
    if world_model_ids.size >= 2**31:
      raise ValueError("Logical world count must fit signed int32 indexing")
    self.world_model_ids = world_model_ids
    self.num_models = num_models
    self.nworld = world_model_ids.size
    with wp.ScopedDevice(self.device):
      self.model_worlds = wp.empty(self.nworld, dtype=int)
      self.model_offsets = wp.zeros(num_models + 1, dtype=int)
      self.model_counts = wp.zeros(num_models, dtype=int)
      self.reset_mask = wp.zeros(self.nworld, dtype=int)
      self.epoch = wp.zeros(self.nworld, dtype=wp.uint64)
      self.transaction_status = wp.zeros(2, dtype=int)
      self.status = wp.zeros(1, dtype=int)
      self._targets = wp.empty(self.nworld, dtype=int)
      self._cursors = wp.zeros(num_models, dtype=int)
      self._changed = wp.zeros(1, dtype=int)
    self._owned_arrays = (
      self.model_worlds,
      self.model_offsets,
      self.model_counts,
      self.reset_mask,
      self.epoch,
      self.transaction_status,
      self.status,
      self._targets,
      self._cursors,
      self._changed,
    )
    self.rebuild()

  @property
  def workspace_bytes(self):
    """Return owned array storage bytes, excluding the bound world assignments."""
    return sum(array.capacity for array in self._owned_arrays)

  def rebuild(self):
    """Rebuild model ranges without allocation; recordable inside a CUDA graph."""
    self.model_counts.zero_()
    self._cursors.zero_()
    wp.launch(_model_histogram, dim=self.nworld, inputs=[self.world_model_ids], outputs=[self.model_counts], device=self.device)
    wp.launch(_prefix_model_counts, dim=1, inputs=[self.model_counts], outputs=[self.model_offsets], device=self.device)
    wp.launch(
      _scatter_worlds,
      dim=self.nworld,
      inputs=[self.world_model_ids, self.model_offsets],
      outputs=[self._cursors, self.model_worlds],
      device=self.device,
    )

  def plan(self, requests):
    """Consume one all-or-none GPU reset transaction and refresh changed worklists."""
    if not isinstance(requests, wp.array) or requests.device != self.device or requests.dtype != wp.int32:
      raise ValueError("requests must be an int32 Warp array on the worklist device")
    if requests.shape != (self.nworld,):
      raise ValueError(f"requests must have shape ({self.nworld},)")
    self.reset_mask.zero_()
    self.transaction_status.zero_()
    self._changed.zero_()
    wp.launch(
      _validate_requests,
      dim=self.nworld,
      inputs=[self.num_models, requests],
      outputs=[self._targets, self.transaction_status],
      device=self.device,
    )
    wp.launch(
      _commit_requests,
      dim=self.nworld,
      inputs=[self._targets, self.transaction_status],
      outputs=[self.world_model_ids, requests, self.reset_mask, self.epoch, self._changed],
      device=self.device,
    )
    if self.device.is_cuda and self.device.is_capturing:
      wp.capture_if(self._changed, self.rebuild)
    else:
      # Avoid host condition evaluation: eager calls unconditionally refresh ranges.
      self.rebuild()
    wp.launch(_record_errors, dim=1, outputs=[self.transaction_status, self.status], device=self.device)

  def clear_status(self):
    """Clear sticky error bits without changing pending requests or assignments."""
    self.status.zero_()

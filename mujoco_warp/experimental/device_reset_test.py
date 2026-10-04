"""Verify GPU world slot planning and captured transactional resets."""

import numpy as np
import pytest
import warp as wp

from mujoco_warp.experimental.device_reset import DeviceResetPlanner

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


def _planner(device, capacities=(1, 1), slots=(0, 1), groups=(0, 1), rows=(0, 0)):
  with wp.ScopedDevice(device):
    offsets = wp.array(np.concatenate(([0], np.cumsum(capacities))).astype(np.int32), dtype=int)
    models = np.arange(len(capacities), dtype=np.int32)
    lookup = np.concatenate((models, [-1])).astype(np.int32)
    return DeviceResetPlanner(
      offsets,
      wp.array(models, dtype=int),
      wp.array(lookup, dtype=int),
      wp.array(slots, dtype=int),
      wp.array(groups, dtype=int),
      wp.array(rows, dtype=int),
      wp.array(models[np.asarray(groups)], dtype=int),
      device=device,
    )


def _assert_consistent(planner):
  slots = planner.slot_world_ids.numpy()
  groups = planner.world_group.numpy()
  rows = planner.world_row.numpy()
  models = planner.world_model_ids.numpy()
  offsets = planner.group_offsets.numpy()
  group_models = planner.group_model_ids.numpy()
  np.testing.assert_array_equal(slots[offsets[groups] + rows], np.arange(planner.nworld))
  np.testing.assert_array_equal(models, group_models[groups])
  np.testing.assert_array_equal(np.sort(slots[slots >= 0]), np.arange(planner.nworld))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_capacity_releasing_transactional_swap_and_same_model_reset(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  planner = _planner(device)
  requests = wp.array([1, 0], dtype=int, device=device)
  planner.plan(requests)
  _assert_consistent(planner)
  np.testing.assert_array_equal(planner.world_group.numpy(), [1, 0])
  np.testing.assert_array_equal(planner.transaction_status.numpy(), [0, 2])
  np.testing.assert_array_equal(planner.reset_mask.numpy(), [1, 1])
  np.testing.assert_array_equal(planner.group_reset_count.numpy(), [1, 1])
  np.testing.assert_array_equal(planner.epoch.numpy(), [1, 1])
  np.testing.assert_array_equal(requests.numpy(), [-1, -1])
  wp.copy(requests, wp.array([1, -1], dtype=int, device=device))
  before_row = planner.world_row.numpy().copy()
  planner.plan(requests)
  _assert_consistent(planner)
  np.testing.assert_array_equal(planner.world_row.numpy(), before_row)
  np.testing.assert_array_equal(planner.transaction_status.numpy(), [0, 1])
  np.testing.assert_array_equal(planner.group_reset_count.numpy(), [0, 1])
  np.testing.assert_array_equal(planner.epoch.numpy(), [2, 1])
  assert planner.workspace_bytes > 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rejected_batch_leaves_all_mappings_slots_requests_and_epochs_unchanged(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  planner = _planner(device, capacities=(2, 2), slots=(0, 1, 2, -1), groups=(0, 0, 1), rows=(0, 1, 0))
  before = {
    name: getattr(planner, name).numpy().copy()
    for name in ("slot_world_ids", "world_group", "world_row", "world_model_ids", "epoch")
  }
  for requested, flag in [([-1, -1, 0], 2), ([2, -1, 0], 1), ([-2, -1, -1], 1)]:
    requests = wp.array(requested, dtype=int, device=device)
    planner.clear_status()
    planner.plan(requests)
    for name, values in before.items():
      np.testing.assert_array_equal(getattr(planner, name).numpy(), values)
    np.testing.assert_array_equal(requests.numpy(), requested)
    assert int(planner.transaction_status.numpy()[0]) & flag
    assert int(planner.transaction_status.numpy()[1]) == 0
    assert int(planner.status.numpy()[0]) & flag
    np.testing.assert_array_equal(planner.reset_mask.numpy(), [0] * 4)
    np.testing.assert_array_equal(planner.group_reset_count.numpy(), [0, 0])
  requests = wp.array([-1, -1, -1], dtype=int, device=device)
  planner.plan(requests)
  np.testing.assert_array_equal(planner.transaction_status.numpy(), [0, 0])
  assert int(planner.status.numpy()[0]) != 0
  planner.clear_status()
  np.testing.assert_array_equal(planner.status.numpy(), [0])


@wp.kernel
def _produce_requests(
  # Out:
  phase_out: wp.array[int],
  requests_out: wp.array[int],
):
  phase = phase_out[0]
  requests_out[0] = -1
  requests_out[1] = -1
  if phase == 0:
    requests_out[0] = 1
    requests_out[1] = 0
  elif phase == 1:
    requests_out[1] = 1
  elif phase == 2:
    requests_out[0] = 0
    requests_out[1] = 1
  phase_out[0] = phase + 1


@wp.kernel
def _record_results(
  # In:
  phase_in: wp.array[int],
  transaction_in: wp.array[int],
  status_in: wp.array[int],
  groups_in: wp.array[int],
  requests_in: wp.array[int],
  epochs_in: wp.array[wp.uint64],
  # Out:
  status_history_out: wp.array2d[int],
  group_history_out: wp.array2d[int],
  request_history_out: wp.array2d[int],
  epoch_history_out: wp.array2d[wp.uint64],
):
  worldid = wp.tid()
  phase = phase_in[0] - 1
  group_history_out[phase, worldid] = groups_in[worldid]
  request_history_out[phase, worldid] = requests_in[worldid]
  epoch_history_out[phase, worldid] = epochs_in[worldid]
  if worldid == 0:
    status_history_out[phase, 0] = transaction_in[0]
    status_history_out[phase, 1] = transaction_in[1]
    status_history_out[phase, 2] = status_in[0]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_captured_gpu_producer_reassigns_overflows_and_recovers_without_host_updates():
  device = "cuda:0"
  planner = _planner(device)
  with wp.ScopedDevice(device):
    requests = wp.full(2, -1, dtype=int)
    phase = wp.zeros(1, dtype=int)
    status_history = wp.zeros((4, 3), dtype=int)
    group_history = wp.zeros((4, 2), dtype=int)
    request_history = wp.zeros((4, 2), dtype=int)
    epoch_history = wp.zeros((4, 2), dtype=wp.uint64)
    planner.plan(requests)
    wp.load_module(module=__name__, device=device)
    pointers = [getattr(planner, name).ptr for name in ("reset_mask", "world_group", "world_row", "epoch")]
    with wp.ScopedCapture() as capture:
      wp.launch(_produce_requests, dim=1, outputs=[phase, requests])
      planner.plan(requests)
      wp.launch(
        _record_results,
        dim=2,
        inputs=[phase, planner.transaction_status, planner.status, planner.world_group, requests, planner.epoch],
        outputs=[status_history, group_history, request_history, epoch_history],
      )
    # No CPU copies, readback, allocation, or synchronization between these replays.
    for _ in range(4):
      wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(status_history.numpy(), [[0, 2, 0], [2, 0, 2], [0, 2, 2], [0, 0, 2]])
    np.testing.assert_array_equal(group_history.numpy(), [[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(request_history.numpy(), [[-1, -1], [-1, 1], [-1, -1], [-1, -1]])
    np.testing.assert_array_equal(epoch_history.numpy(), [[1, 1], [1, 1], [2, 2], [2, 2]])
    assert pointers == [getattr(planner, name).ptr for name in ("reset_mask", "world_group", "world_row", "epoch")]
  _assert_consistent(planner)

"""Verify capacity-free model worklists and GPU reset transactions."""

import numpy as np
import pytest
import warp as wp

from mujoco_warp.experimental.pooled_worklist import ModelWorklist

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


def _check_worklist(worklist):
  models = worklist.world_model_ids.numpy()
  worlds = worklist.model_worlds.numpy()
  offsets = worklist.model_offsets.numpy()
  counts = worklist.model_counts.numpy()
  np.testing.assert_array_equal(np.sort(worlds), np.arange(worklist.nworld))
  np.testing.assert_array_equal(np.diff(offsets), counts)
  assert offsets[0] == 0 and offsets[-1] == worklist.nworld
  for model in range(worklist.num_models):
    np.testing.assert_array_equal(np.sort(worlds[offsets[model] : offsets[model + 1]]), np.flatnonzero(models == model))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_every_world_can_use_any_catalog_model_without_occupancy_limits(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  worlds = 8192
  assignments = wp.array(np.arange(worlds, dtype=np.int32) % 3, dtype=int, device=device)
  worklist = ModelWorklist(assignments, 3, device=device)
  for model in [2, 0, 1]:
    requests = wp.full(worlds, model, dtype=int, device=device)
    worklist.plan(requests)
    _check_worklist(worklist)
    expected = [0, 0, 0]
    expected[model] = worlds
    np.testing.assert_array_equal(worklist.model_counts.numpy(), expected)
    np.testing.assert_array_equal(worklist.world_model_ids.numpy(), np.full(worlds, model))
    np.testing.assert_array_equal(requests.numpy(), np.full(worlds, -1))
    np.testing.assert_array_equal(worklist.transaction_status.numpy(), [0, worlds])
    np.testing.assert_array_equal(worklist.reset_mask.numpy(), np.ones(worlds))
  np.testing.assert_array_equal(worklist.epoch.numpy(), np.full(worlds, 3))
  assert worklist.workspace_bytes < worlds * 24


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_invalid_model_rejects_entire_batch_and_retains_requests(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  assignments = wp.array([0, 1, 2], dtype=int, device=device)
  worklist = ModelWorklist(assignments, 3, device=device)
  requests = wp.array([1, 99, 0], dtype=int, device=device)
  worklist.plan(requests)
  np.testing.assert_array_equal(assignments.numpy(), [0, 1, 2])
  np.testing.assert_array_equal(requests.numpy(), [1, 99, 0])
  np.testing.assert_array_equal(worklist.epoch.numpy(), [0, 0, 0])
  np.testing.assert_array_equal(worklist.reset_mask.numpy(), [0, 0, 0])
  np.testing.assert_array_equal(worklist.transaction_status.numpy(), [1, 0])
  np.testing.assert_array_equal(worklist.status.numpy(), [1])
  _check_worklist(worklist)
  worklist.clear_status()
  np.testing.assert_array_equal(worklist.status.numpy(), [0])


@wp.kernel
def _produce_model_requests(
  # Out:
  phase_out: wp.array[int],
  requests_out: wp.array[int],
):
  phase = phase_out[0]
  for worldid in range(4):
    requests_out[worldid] = -1
  if phase == 0:
    for worldid in range(4):
      requests_out[worldid] = 2
  elif phase == 1:
    requests_out[0] = 0
    requests_out[1] = 99
  elif phase == 2:
    requests_out[0] = 0
    requests_out[1] = 1
    requests_out[3] = 0
  elif phase == 3:
    requests_out[0] = 0
    requests_out[1] = 1
    requests_out[2] = 2
    requests_out[3] = 0
  phase_out[0] = phase + 1


@wp.kernel
def _record_model_results(
  # In:
  phase_in: wp.array[int],
  assignments_in: wp.array[int],
  counts_in: wp.array[int],
  worlds_in: wp.array[int],
  epochs_in: wp.array[wp.uint64],
  reset_mask_in: wp.array[int],
  transaction_in: wp.array[int],
  status_in: wp.array[int],
  # Out:
  model_history_out: wp.array2d[int],
  count_history_out: wp.array2d[int],
  world_history_out: wp.array2d[int],
  epoch_history_out: wp.array2d[wp.uint64],
  reset_history_out: wp.array2d[int],
  status_history_out: wp.array2d[int],
):
  worldid = wp.tid()
  phase = phase_in[0] - 1
  model_history_out[phase, worldid] = assignments_in[worldid]
  world_history_out[phase, worldid] = worlds_in[worldid]
  epoch_history_out[phase, worldid] = epochs_in[worldid]
  reset_history_out[phase, worldid] = reset_mask_in[worldid]
  if worldid == 0:
    for modelid in range(counts_in.shape[0]):
      count_history_out[phase, modelid] = counts_in[modelid]
    status_history_out[phase, 0] = transaction_in[0]
    status_history_out[phase, 1] = transaction_in[1]
    status_history_out[phase, 2] = status_in[0]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_gpu_generated_requests_rebuild_worklists_only_when_assignments_change():
  with wp.ScopedDevice("cuda:0"):
    assignments = wp.array([0, 1, 2, 0], dtype=int)
    worklist = ModelWorklist(assignments, 3)
    requests = wp.full(4, -1, dtype=int)
    phase = wp.zeros(1, dtype=int)
    models = wp.zeros((5, 4), dtype=int)
    counts = wp.zeros((5, 3), dtype=int)
    worlds = wp.zeros((5, 4), dtype=int)
    epochs = wp.zeros((5, 4), dtype=wp.uint64)
    resets = wp.zeros((5, 4), dtype=int)
    statuses = wp.zeros((5, 3), dtype=int)
    worklist.plan(requests)
    wp.load_module(module=__name__, device="cuda:0")
    pointers = [array.ptr for array in worklist._owned_arrays]
    with wp.ScopedCapture() as capture:
      wp.launch(_produce_model_requests, dim=1, outputs=[phase, requests])
      worklist.plan(requests)
      wp.launch(
        _record_model_results,
        dim=4,
        inputs=[
          phase,
          assignments,
          worklist.model_counts,
          worklist.model_worlds,
          worklist.epoch,
          worklist.reset_mask,
          worklist.transaction_status,
          worklist.status,
        ],
        outputs=[models, counts, worlds, epochs, resets, statuses],
      )
    for _ in range(5):
      wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(models.numpy(), [[2, 2, 2, 2], [2, 2, 2, 2], [0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]])
    np.testing.assert_array_equal(counts.numpy(), [[0, 0, 4], [0, 0, 4], [2, 1, 1], [2, 1, 1], [2, 1, 1]])
    np.testing.assert_array_equal(epochs.numpy(), [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 1, 2], [3, 3, 2, 3], [3, 3, 2, 3]])
    np.testing.assert_array_equal(resets.numpy(), [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    np.testing.assert_array_equal(statuses.numpy(), [[0, 4, 0], [1, 0, 1], [0, 3, 1], [0, 4, 1], [0, 0, 1]])
    # Invalid transactions, same-model resets, and no-op steps retain the cached list exactly.
    world_rows = worlds.numpy()
    np.testing.assert_array_equal(world_rows[0], world_rows[1])
    np.testing.assert_array_equal(world_rows[2], world_rows[3])
    np.testing.assert_array_equal(world_rows[3], world_rows[4])
    assert pointers == [array.ptr for array in worklist._owned_arrays]
  _check_worklist(worklist)

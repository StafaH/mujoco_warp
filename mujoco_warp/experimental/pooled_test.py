"""GPU-only population changes over a shared homogeneous physics workspace."""

import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental.heterogeneous_test import _models
from mujoco_warp.experimental.pooled import PooledHeterogeneousWorlds

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _population_changes(
  # In:
  tick: wp.array[int],
  # Out:
  requests_out: wp.array[int],
):
  world = wp.tid()
  if tick[0] == 1:
    requests_out[world] = 1
  elif tick[0] == 2 and world == 0:
    requests_out[world] = 0
  elif tick[0] == 3:
    requests_out[world] = 0
  elif tick[0] == 5 and world < 2:
    requests_out[world] = world % 2


@wp.kernel
def _advance_tick(
  # Data out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


@wp.kernel
def _controls(
  # Data out:
  ctrl_out: wp.array2d[float],
):
  world, actuator = wp.tid()
  ctrl_out[world, actuator] = 0.1 * float(world + 1)


@wp.kernel
def _record_previous_state(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  tick: wp.array[int],
  # Out:
  snapshots_out: wp.array3d[float],
):
  world, joint = wp.tid()
  if tick[0] > 0:
    snapshots_out[tick[0] - 1, world, joint] = qpos_in[world, joint]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [2, 8])
def test_arbitrary_population_changes_under_one_graph(batch_size, monkeypatch):
  models = _models()
  initial = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
  worlds = PooledHeterogeneousWorlds(models, initial, batch_size=batch_size, device="cuda:0", scratch_bytes=8 << 20)
  tick = wp.zeros(1, dtype=int, device="cuda:0")
  snapshots = wp.zeros((6, 7, 2), dtype=float, device="cuda:0")

  def episodes(batch):
    wp.launch(_record_previous_state, dim=(7, 2), inputs=[batch.state.qpos, tick], outputs=[snapshots], device=batch.device)
    wp.launch(_population_changes, dim=7, inputs=[tick], outputs=[batch.reset_model_ids], device=batch.device)
    wp.launch(_advance_tick, dim=1, outputs=[tick], device=batch.device)

  def actions(batch):
    wp.launch(_controls, dim=batch.state.ctrl.shape, outputs=[batch.state.ctrl], device=batch.device)

  def forbidden(*args, **kwargs):
    raise AssertionError("No host readback or synchronization in the captured episode loop")

  with monkeypatch.context() as patch:
    patch.setattr(wp.array, "numpy", forbidden)
    patch.setattr(wp, "synchronize_device", forbidden)
    worlds.capture(steps_per_launch=2, before_reset=episodes, before_step=actions)
    graph = worlds.graph
    pointer = worlds.state.qpos.ptr
    worlds.step()
    worlds.step()
    worlds.step()
  assert worlds.graph is graph
  assert worlds.state.qpos.ptr == pointer
  assignments = initial.copy()
  actual_snapshots = snapshots.numpy()
  with wp.ScopedDevice("cuda:0"):
    device_models = [io.put_model(model) for model in models]
    reference = [io.make_data(models[model], nworld=1) for model in assignments]
    for step in range(6):
      reset = np.full(7, -1, dtype=np.int32)
      if step == 1:
        reset[:] = 1
      elif step == 2:
        reset[0] = 0
      elif step == 3:
        reset[:] = 0
      elif step == 5:
        reset[:2] = [0, 1]
      for world in range(7):
        if reset[world] >= 0:
          assignments[world] = reset[world]
          reference[world] = io.make_data(models[reset[world]], nworld=1)
        reference[world].ctrl.fill_(0.1 * (world + 1))
        forward.step(device_models[assignments[world]], reference[world])
        if step < 5:
          np.testing.assert_allclose(
            actual_snapshots[step, world, : models[assignments[world]].nq],
            reference[world].qpos.numpy()[0],
            atol=1e-7,
            rtol=1e-6,
          )
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), assignments)
  np.testing.assert_array_equal(worlds.reset_model_ids.numpy(), -1)
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [4, 3, 2, 2, 2, 2, 2])
  assert worlds.device_reset_status.numpy()[0] == 0
  for world, model_id in enumerate(assignments):
    for name, width in (("qpos", models[model_id].nq), ("qvel", models[model_id].nv), ("time", 1)):
      expected = getattr(reference[world], name).numpy().reshape(-1)
      actual = getattr(worlds.state, name).numpy()[world].reshape(-1)[:width]
      np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_invalid_request_is_atomic_across_pooled_worlds():
  worlds = PooledHeterogeneousWorlds(_models(), [0, 1], batch_size=1, device="cuda:0", scratch_bytes=8 << 20)
  worlds.reset_model_ids.assign(np.array([1, 9], dtype=np.int32))
  worlds.capture()
  worlds.step()
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [0, 1])
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [0, 0])
  np.testing.assert_array_equal(worlds.reset_model_ids.numpy(), [1, 9])
  assert worlds.device_reset_status.numpy()[0] != 0
  np.testing.assert_allclose(worlds.state.time.numpy().reshape(-1), [0.002, 0.004])


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_shared_workspace_memory_does_not_scale_with_model_reservations():
  models = _models()
  worlds = PooledHeterogeneousWorlds(models, [0, 1] * 8, batch_size=4, device="cuda:0", scratch_bytes=8 << 20)
  worlds.capture()
  before = worlds.memory_report()
  worlds.reset_model_ids.fill_(1)
  worlds.step()
  after = worlds.memory_report()
  assert after["total_bytes"] == before["total_bytes"]
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), np.ones(16, dtype=np.int32))
  assert worlds.state.qpos.shape[0] == 16
  assert worlds.device_reset_status.numpy()[0] == 0

"""GPU episode transitions execute inside the same physics graph."""

import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental import HeterogeneousWorlds
from mujoco_warp.experimental.heterogeneous_test import _models

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _episode_requests(
  # In:
  tick: wp.array[int],
  model_ids: wp.array[int],
  # Out:
  requests_out: wp.array[int],
):
  world = wp.tid()
  if (tick[0] == 1 and world < 2) or (tick[0] == 3 and world >= 2):
    requests_out[world] = 1 - model_ids[world]


@wp.kernel
def _tick(
  # Data out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


@wp.kernel
def _actions(
  # In:
  world_ids: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
):
  row, actuator = wp.tid()
  world = world_ids[row]
  if world >= 0:
    ctrl_out[row, actuator] = 0.1 * float(world + 1)


def _apply_actions(worlds):
  for group in worlds.groups:
    wp.launch(_actions, dim=group.data.ctrl.shape, inputs=[group.world_ids], outputs=[group.data.ctrl], device=worlds.device)


def _world_values(worlds, world, field):
  """Explicit readback for assertions after the captured workload finishes."""
  group = worlds.groups[int(worlds.world_group.numpy()[world])]
  row = int(worlds.world_row.numpy()[world])
  return getattr(group.data, field).numpy()[row]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
@pytest.mark.parametrize("memory", ["default", "virtual"])
def test_gpu_generated_model_changes_reuse_graph_and_match_reference(memory, monkeypatch):
  models = _models()
  worlds = HeterogeneousWorlds(models, [0, 1, 0, 1], device="cuda:0", memory=memory)
  reference_ids = [0, 1, 0, 1]
  with wp.ScopedDevice("cuda:0"):
    reference_models = [io.put_model(model) for model in models]
    reference_data = [io.make_data(models[model], nworld=1) for model in reference_ids]
  requests = worlds.enable_device_resets()
  tick = wp.zeros(1, dtype=int, device=worlds.device)

  def episode_logic(batch):
    wp.launch(_episode_requests, dim=4, inputs=[tick, batch.world_model_ids], outputs=[requests], device=batch.device)
    wp.launch(_tick, dim=1, outputs=[tick], device=batch.device)

  def forbidden_host_operation(*args, **kwargs):
    raise AssertionError("The captured episode loop must not read or synchronize with the host")

  with monkeypatch.context() as patch:
    patch.setattr(wp.array, "numpy", forbidden_host_operation)
    patch.setattr(wp, "synchronize_device", forbidden_host_operation)
    worlds.capture(steps_per_launch=2, before_reset=episode_logic, before_step=_apply_actions)
    graph = worlds.graph
    pointers = [group.data.qpos.ptr for group in worlds.groups]
    worlds.step()
    worlds.step()
  with wp.ScopedDevice("cuda:0"):
    for step in range(4):
      selected = [0, 1] if step == 1 else [2, 3] if step == 3 else []
      for world in selected:
        reference_ids[world] = 1 - reference_ids[world]
        reference_data[world] = io.make_data(models[reference_ids[world]], nworld=1)
      for world, data in enumerate(reference_data):
        data.ctrl.fill_(0.1 * (world + 1))
        forward.step(reference_models[reference_ids[world]], data)
  assert worlds.graph is graph
  assert [group.data.qpos.ptr for group in worlds.groups] == pointers
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [1, 0, 1, 0])
  np.testing.assert_array_equal(requests.numpy(), -1)
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [1, 1, 1, 1])
  assert worlds.device_reset_status.numpy()[0] == 0
  for world in range(4):
    for field in ("qpos", "qvel", "ctrl", "time"):
      np.testing.assert_allclose(
        _world_values(worlds, world, field), getattr(reference_data[world], field).numpy()[0], atol=1e-7
      )
  for query in (lambda: worlds.locate(0), lambda: worlds.world_array(0, "qpos")):
    with pytest.raises(RuntimeError):
      query()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_gpu_overflow_rejects_entire_transition_and_retains_requests():
  worlds = HeterogeneousWorlds(_models(), [0, 1], device="cuda:0")
  reference = HeterogeneousWorlds(_models(), [0, 1], device="cuda:0")
  requests = worlds.enable_device_resets()
  requests.assign(np.array([1, -1], dtype=np.int32))
  worlds.capture()
  worlds.step()
  reference.step()
  assert worlds.device_reset_status.numpy()[0] != 0
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [0, 1])
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [0, 0])
  np.testing.assert_array_equal(requests.numpy(), [1, -1])
  for world in range(2):
    for field in ("qpos", "qvel", "time"):
      np.testing.assert_allclose(_world_values(worlds, world, field), reference.world_array(world, field).numpy()[0], atol=1e-7)
  # Supply the matching departure; the previously rejected request can now commit.
  requests.assign(np.array([1, 0], dtype=np.int32))
  worlds.step()
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [1, 0])
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [1, 1])
  assert worlds.device_reset_status.numpy()[0] != 0  # Sticky diagnostic survives successful retry.


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_gpu_switch_can_activate_initially_unused_reserved_model():
  worlds = HeterogeneousWorlds(_models(), [0, 0], device="cuda:0", capacity_per_model={0: 2, 1: 2})
  requests = worlds.enable_device_resets()
  requests.assign(np.array([1, 1], dtype=np.int32))
  worlds.capture(before_step=_apply_actions)
  worlds.step()
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [1, 1])
  np.testing.assert_array_equal(worlds.groups[0].world_ids.numpy(), [-1, -1])
  np.testing.assert_array_equal(np.sort(worlds.groups[1].world_ids.numpy()), [0, 1])
  for world in range(2):
    assert _world_values(worlds, world, "qpos").shape == (2,)
    np.testing.assert_allclose(_world_values(worlds, world, "ctrl"), 0.1 * (world + 1))
    np.testing.assert_allclose(_world_values(worlds, world, "time"), 0.004)
  report = worlds.memory_report()
  assert [group["active_worlds"] for group in report["groups"]] == [0, 2]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_gpu_invalid_request_cannot_partially_publish_other_changes():
  worlds = HeterogeneousWorlds(_models(), [0, 1], device="cuda:0", capacity_per_model={0: 2, 1: 2})
  requests = worlds.enable_device_resets()
  requests.assign(np.array([1, 2], dtype=np.int32))
  worlds.capture()
  worlds.step()
  assert worlds.device_reset_status.numpy()[0] != 0
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [0, 1])
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [0, 0])
  np.testing.assert_array_equal(requests.numpy(), [1, 2])

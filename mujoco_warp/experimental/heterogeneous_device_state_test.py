"""GPU model changes preserve ongoing contacts, activation history, and sleep state."""

import mujoco
import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental import HeterogeneousWorlds
from mujoco_warp.experimental.heterogeneous_device_reset_test import _apply_actions
from mujoco_warp.experimental.heterogeneous_device_reset_test import _world_values
from mujoco_warp.experimental.heterogeneous_test import _models

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _request_after_warmup(
  # In:
  tick: wp.array[int],
  # Out:
  requests_out: wp.array[int],
):
  if tick[0] == 20:
    requests_out[0] = 1


@wp.kernel
def _advance_tick(
  # Out:
  counter_out: wp.array[int],
):
  counter_out[0] += 1


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_gpu_switch_preserves_other_world_contacts_history_activation_and_sleep():
  contact_model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <option timestep=".002" sleep_tolerance=".01"><flag sleep="enable" island="enable"/></option>
      <worldbody><geom type="plane" size="5 5 .1"/>
        <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
        <body pos="1 0 1"><joint name="drive" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".05" mass="1"/></body>
      </worldbody>
      <actuator><general joint="drive" dyntype="filter" dynprm=".05"
        delay=".004" nsample="3" gainprm="1"/></actuator>
      <sensor><jointpos joint="drive" delay=".004" nsample="3"/></sensor>
    </mujoco>
  """)
  models = [contact_model, _models()[0]]
  worlds = HeterogeneousWorlds(models, [0, 0], device="cuda:0", capacity_per_model={0: 3, 1: 1})
  requests = worlds.enable_device_resets()
  tick = wp.zeros(1, dtype=int, device=worlds.device)

  def episode_logic(batch):
    wp.launch(_request_after_warmup, dim=1, inputs=[tick], outputs=[requests], device=batch.device)
    wp.launch(_advance_tick, dim=1, outputs=[tick], device=batch.device)

  with wp.ScopedDevice(worlds.device):
    reference_models = [io.put_model(model) for model in models]
    reference_data = [io.make_data(contact_model), io.make_data(contact_model)]

  def step_reference():
    with wp.ScopedDevice(worlds.device):
      for world, data in enumerate(reference_data):
        data.ctrl.fill_(0.1 * (world + 1))
        forward.step(reference_models[1 if tick_value > 20 and world == 0 else 0], data)

  worlds.capture(before_reset=episode_logic, before_step=_apply_actions)
  graph = worlds.graph
  tick_value = 0
  for _ in range(20):
    worlds.step()
    step_reference()
    tick_value += 1
  assert _world_values(worlds, 1, "act")[0] > 0
  assert np.any(_world_values(worlds, 1, "history") != 0)
  assert _world_values(worlds, 1, "tree_asleep")[0] >= 0

  # A new independent Data object is the reference episode reset; the actual transition
  # is generated inside the captured GPU graph without a host reset or graph change.
  with wp.ScopedDevice(worlds.device):
    reference_data[0] = io.make_data(models[1])
  tick_value = 21
  for _ in range(4):
    worlds.step()
    step_reference()
  assert worlds.graph is graph
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), [1, 0])
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [1, 0])
  for world, data in enumerate(reference_data):
    for name in ("qpos", "qvel", "act", "history", "qacc_warmstart", "sensordata", "tree_asleep", "tree_awake"):
      np.testing.assert_allclose(_world_values(worlds, world, name), getattr(data, name).numpy()[0], atol=1e-6, err_msg=name)
  group = worlds.groups[int(worlds.world_group.numpy()[1])]
  row = int(worlds.world_row.numpy()[1])
  count = int(group.data.nacon.numpy()[0])
  # Asleep trees omit their contacts; compare the surviving world's independent run.
  assert np.count_nonzero(group.data.contact.worldid.numpy()[:count] == row) == int(reference_data[1].nacon.numpy()[0])

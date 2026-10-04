"""Compare pooled chunk state with independent worlds through model changes."""

import mujoco
import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental.pooled import PooledHeterogeneousWorlds

wp.set_module_options({"enable_backward": False})


def _history_contact_models(integrator):
  models = []
  for catalog_id in range(2):
    x, y, height = (1, 2, 0.03) if catalog_id == 0 else (3, -2, 0.01)
    extra = '<body pos="0 0 3"><joint type="slide"/><geom size="0.05" mass="1"/></body>' if catalog_id else ""
    models.append(
      mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <size nuserdata="2"/>
          <option timestep="0.002" integrator="{integrator}" solver="Newton" iterations="15"/>
          <worldbody>
            <geom type="plane" pos="{x} {y} {height}" euler="0.1 0.05 0" size="5 5 0.1"/>
            <body name="ball" pos="{x} {y} {height + 0.085}">
              <freejoint/><geom size="0.1" mass="1"/><site name="imu"/>
            </body>
            <body pos="0 0 2">
              <joint name="slider" type="slide" axis="1 0 0" damping="0.1"/>
              <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
            </body>
            <body name="target" mocap="true" pos="0 0 4"><site/></body>
            {extra}
          </worldbody>
          <actuator>
            <general joint="slider" dyntype="filter" dynprm="0.02" delay="0.004" nsample="5" interp="linear"/>
          </actuator>
          <sensor>
            <jointpos joint="slider" delay="0.004" nsample="5" interp="linear"/>
            <framepos objtype="body" objname="target"/>
            <accelerometer site="imu"/>
          </sensor>
        </mujoco>
      """)
    )
  return models


@wp.kernel
def _reset_one_world(
  # In:
  tick: wp.array[int],
  # Out:
  requests_out: wp.array[int],
  tick_out: wp.array[int],
):
  if tick[0] == 3:
    requests_out[1] = 0
  tick_out[0] += 1


@wp.kernel
def _persistent_inputs(
  # In:
  tick: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
  qfrc_applied_out: wp.array2d[float],
  mocap_pos_out: wp.array2d[wp.vec3],
  userdata_out: wp.array2d[float],
):
  world = wp.tid()
  t = float(tick[0])
  ctrl_out[world, 0] = 0.2 * float(world + 1) + 0.01 * t
  userdata_out[world, 0] = float(world + 1)
  userdata_out[world, 1] = t
  mocap_pos_out[world, 0] = wp.vec3(float(world), 0.01 * t, 4.0)
  for dof in range(qfrc_applied_out.shape[1]):
    qfrc_applied_out[world, dof] = 0.02 * float(world + 1)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
@pytest.mark.parametrize("integrator", ["Euler", "RK4", "implicitfast", "implicit"])
def test_chunked_persistent_state_matches_independent_worlds(integrator):
  models = _history_contact_models(integrator)
  assignments = np.array([0, 1, 0, 1, 0], dtype=np.int32)
  worlds = PooledHeterogeneousWorlds(models, assignments, batch_size=2, device="cuda:0", scratch_bytes=16 << 20)
  tick = wp.zeros(1, dtype=int, device=worlds.device)

  def resets(batch):
    wp.launch(_reset_one_world, dim=1, inputs=[tick], outputs=[batch.reset_model_ids, tick], device=batch.device)

  def actions(batch):
    wp.launch(
      _persistent_inputs,
      dim=batch.nworld,
      inputs=[tick],
      outputs=[batch.state.ctrl, batch.state.qfrc_applied, batch.state.mocap_pos, batch.state.userdata],
      device=batch.device,
    )

  worlds.capture(before_reset=resets, before_step=actions)
  with wp.ScopedDevice(worlds.device):
    device_models = [io.put_model(model) for model in models]
    references = [io.make_data(models[model], nworld=1) for model in assignments]
    for model_id, data in zip(assignments, references, strict=True):
      io.reset_data(device_models[model_id], data)
    for step_id in range(8):
      worlds.step()
      if step_id == 3:
        assignments[1] = 0
        references[1] = io.make_data(models[0], nworld=1)
        io.reset_data(device_models[0], references[1])
      for world_id, data in enumerate(references):
        data.ctrl.fill_(0.2 * (world_id + 1) + 0.01 * (step_id + 1))
        data.userdata.assign(np.array([[world_id + 1, step_id + 1]], dtype=np.float32))
        data.mocap_pos.assign(np.array([[[world_id, 0.01 * (step_id + 1), 4]]], dtype=np.float32))
        data.qfrc_applied.fill_(0.02 * (world_id + 1))
        forward.step(device_models[assignments[world_id]], data)
      for field in ("qpos", "qvel", "act", "history", "sensordata", "userdata", "mocap_pos", "qacc_warmstart", "time"):
        actual = getattr(worlds.state, field).numpy()
        for world_id, data in enumerate(references):
          expected = getattr(data, field).numpy()[0].reshape(-1)
          observed = actual[world_id].reshape(-1)[: expected.size]
          np.testing.assert_allclose(
            observed,
            expected,
            atol=3e-5,
            rtol=3e-4,
            err_msg=f"{integrator} step {step_id}, world {world_id}, field {field}",
          )
  np.testing.assert_array_equal(worlds.world_model_ids.numpy(), assignments)
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), [0, 1, 0, 0, 0])
  assert worlds.device_reset_status.numpy()[0] == 0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_dense_padded_solver_survives_reused_workspace():
  models = []
  for hinge_count in (27, 29):
    bodies = "".join(
      '<body pos="0 0 2"><joint type="hinge" damping="0.1" limited="true" range="0.1 0.2"/>'
      '<geom size="0.05" mass="1" contype="0" conaffinity="0"/></body>'
      for _ in range(hinge_count)
    )
    models.append(
      mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <option timestep="0.002" solver="Newton" jacobian="dense" iterations="10"/>
          <worldbody>
            <geom type="plane" size="5 5 0.1"/>
            <body pos="0 0 0.09"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
            {bodies}
          </worldbody>
        </mujoco>
      """)
    )
  assert [model.nv for model in models] == [33, 35]
  assignments = [0, 1, 0]
  worlds = PooledHeterogeneousWorlds(models, assignments, batch_size=2, device="cuda:0", scratch_bytes=32 << 20)
  worlds.state.qfrc_applied.fill_(0.01)
  worlds.capture()
  with wp.ScopedDevice(worlds.device):
    device_models = [io.put_model(model) for model in models]
    references = [io.make_data(models[model], nworld=1) for model in assignments]
    for data in references:
      data.qfrc_applied.fill_(0.01)
    for step_id in range(3):
      # An unrelated aliased layout may leave arbitrary bits, including NaNs, in
      # Jacobian padding. Regular make_data retains zeros there for its lifetime.
      worlds._groups[0].data.efc.J.fill_(wp.nan)
      worlds.step()
      for world_id, data in enumerate(references):
        forward.step(device_models[assignments[world_id]], data)
      for field in ("qpos", "qvel", "qacc_warmstart"):
        actual = getattr(worlds.state, field).numpy()
        for world_id, data in enumerate(references):
          expected = getattr(data, field).numpy()[0]
          np.testing.assert_allclose(
            actual[world_id, : expected.size],
            expected,
            atol=3e-5,
            rtol=3e-4,
            err_msg=f"Dense padded solve step {step_id}, world {world_id}, field {field}",
          )

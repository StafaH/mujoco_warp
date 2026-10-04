"""Shared native contact columns drive collision, constraints, and sensors directly."""

import dataclasses

import mujoco
import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental import HeterogeneousWorlds
from mujoco_warp.experimental import allocated_bytes
from mujoco_warp.experimental.native_contact_table import NativeContactTable


@wp.kernel
def _record_count(
  # In:
  count: wp.array[int],
  slot: int,
  # Out:
  result_out: wp.array[int],
):
  result_out[slot] = count[0]


def _model(condim, cone, objects=1, sleeping=False):
  bodies = "".join(
    f'<body pos="{index} 0 .09"><freejoint/><geom name="object{index}" type="sphere" size=".1"/>'
    f'<site name="touch{index}" type="sphere" size=".15"/></body>'
    for index in range(objects)
  )
  flags = '<flag sleep="enable" island="enable"/>' if sleeping else ""
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option cone="{cone}" iterations="8" sleep_tolerance=".01">{flags}</option>
      <default><geom condim="{condim}" friction=".7 .03 .02"/></default>
      <worldbody><geom name="floor" type="plane" size="5 5 .1"/>{bodies}</worldbody>
      <sensor><touch site="touch0"/><distance geom1="floor" geom2="object0" cutoff="1"/>
        <contact geom1="floor" geom2="object0" data="found force dist pos normal"/>
      </sensor>
    </mujoco>
  """)


def _groups(pool, cone, sleeping=False):
  result = []
  for condim, objects, worlds in ((1, 1, 2), (6, 2, 1)):
    mjm = _model(condim, cone, objects, sleeping)
    model = io.put_model(mjm)
    native = io.make_data(mjm, nworld=worlds, naconmax=16, njmax=40)
    shared = io.make_data(mjm, nworld=worlds, naconmax=16, njmax=40, contact_storage=pool)
    result.append((model, native, shared))
  return result


def _assert_state(native, shared):
  for field in ("qpos", "qvel", "qacc", "qfrc_constraint", "sensordata", "nefc", "tree_awake"):
    np.testing.assert_allclose(getattr(shared, field).numpy(), getattr(native, field).numpy(), atol=2e-5, rtol=2e-5)


def _assert_contacts(native, shared):
  count = int(native.nacon.numpy()[0])
  assert int(shared.nacon.numpy()[0]) == count
  # Native rows have no ordering guarantee. Sort by world and geometry, then position.
  fields = ("worldid", "geom", "pos", "dist", "dim", "friction", "type", "adhesion")
  if native.contact.flex.size:
    fields += ("flex", "elem", "vert")

  def rows(data):
    columns = [getattr(data.contact, name).numpy()[:count].reshape(count, -1) for name in fields]
    return sorted(np.concatenate(columns, axis=1).tolist())

  if count:
    np.testing.assert_allclose(rows(shared), rows(native), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_shared_native_contacts_drive_sequential_steps_and_sensors(device, cone):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  with wp.ScopedDevice(device):
    pool = NativeContactTable(16, device=device, memory="default")
    groups = _groups(pool, cone)
    assert groups[0][2].contact.pos.ptr == groups[1][2].contact.pos.ptr
    assert groups[1][2].contact.efc_address.shape[1] >= 10
    assert all(group[2].contact.flex.size == 0 for group in groups)
    for _ in range(5):
      for model, native, shared in groups:
        forward.step(model, native)
        forward.step(model, shared)
        pool.observe(shared)
        _assert_state(native, shared)
        _assert_contacts(native, shared)
    pool.check_errors()


def test_shared_native_contacts_preserve_sleep_processing():
  with wp.ScopedDevice("cpu"):
    pool = NativeContactTable(16, device="cpu", memory="default")
    groups = _groups(pool, "pyramidal", sleeping=True)
    for _ in range(18):
      for model, native, shared in groups:
        forward.step(model, native)
        forward.step(model, shared)
        _assert_state(native, shared)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_shared_native_contacts_preserve_flex_identifiers(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option solver="CG" timestep=".001"/>
      <worldbody><geom type="plane" size="5 5 .1"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1"
                  pos="0 0 .01" radius=".02" dim="2" mass=".5">
          <contact condim="3" selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """)
  with wp.ScopedDevice(device):
    pool = NativeContactTable(64, include_flex=True, device=device, memory="default")
    warped = io.put_model(model)
    native = io.make_data(model, naconmax=64, njmax=100)
    shared = io.make_data(model, naconmax=64, njmax=100, contact_storage=pool)
    assert shared.contact.flex.shape == (64,)
    for _ in range(3):
      forward.step(warped, native)
      forward.step(warped, shared)
      assert int(shared.nacon.numpy()[0]) > 0
      _assert_state(native, shared)
      _assert_contacts(native, shared)


def test_shared_native_overflow_is_sticky_across_groups():
  with wp.ScopedDevice("cpu"):
    pool = NativeContactTable(16, device="cpu", memory="default")
    crowded = _model(3, "pyramidal", objects=2)
    data = io.make_data(crowded, naconmax=1, contact_storage=pool)
    forward.forward(io.put_model(crowded), data)
    assert int(data.ncollision.numpy()[0]) > data.naconmax
    pool.observe(data)
    quiet = _model(1, "pyramidal")
    other = io.make_data(quiet, naconmax=16, contact_storage=pool)
    forward.forward(io.put_model(quiet), other)
    pool.observe(other)
    with pytest.raises(RuntimeError, match="capacity|overflow|exhausted"):
      pool.check_errors()


def test_shared_native_overflow_reports_narrowphase_and_can_clear_errors():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>
  """)
  with wp.ScopedDevice("cpu"):
    pool = NativeContactTable(16, device="cpu", memory="default")
    data = io.make_data(model, naconmax=1, contact_storage=pool)
    forward.forward(io.put_model(model), data)
    assert int(data.nacon.numpy()[0]) > data.naconmax
    pool.observe(data)
    with pytest.raises(RuntimeError, match="capacity|overflow|exhausted"):
      pool.check_errors()
    pool.clear_errors()
    pool.check_errors()


def test_native_contact_views_preserve_model_shapes_and_validate_capacity():
  model = _model(1, "pyramidal")
  pool = NativeContactTable(16, include_flex=True, device="cpu", memory="default")
  contact = pool.make_contact(model, 4, 1)
  assert contact.pos.shape == (4,)
  assert contact.efc_address.shape == (4, 1)
  assert all(getattr(contact, name).size == 0 for name in ("flex", "elem", "vert"))
  assert contact.efc_address.strides == pool.columns["efc_address"].strides
  assert np.all(contact.efc_address.numpy() == -1)
  assert pool.allocated_bytes >= sum(column.capacity for column in pool.columns.values())
  with pytest.raises(ValueError, match="capacity"):
    pool.make_contact(model, 17, 1)
  with pytest.raises(ValueError, match="pyramid"):
    pool.make_contact(model, 4, 11)
  warped = io.put_model(model)
  warped.opt.run_collision_detection = False
  with pytest.raises(ValueError, match="run_collision_detection"):
    pool.validate_model(warped)


def test_heterogeneous_native_pool_counts_storage_once_and_rejects_persistent_input():
  models = [_model(1, "pyramidal"), _model(6, "pyramidal", objects=2)]
  options = {index: {"naconmax": 16, "njmax": 40} for index in range(2)}
  pool = NativeContactTable(16, device="cpu", memory="default")
  native = HeterogeneousWorlds(models, [0, 0, 1], device="cpu", data_kwargs=options)
  shared = HeterogeneousWorlds(models, [0, 0, 1], device="cpu", data_kwargs=options, contact_storage=pool)
  native_report = native.memory_report()
  report = shared.memory_report()
  native_contact_bytes = sum(allocated_bytes(group.data.contact) for group in native.groups)
  assert report["shared_contact_bytes"] == pool.allocated_bytes
  assert report["total_bytes"] == native_report["total_bytes"] - native_contact_bytes + pool.allocated_bytes
  with pytest.raises(RuntimeError, match="transient"):
    shared.arrays("contact.pos")
  shared.step()
  native.step()
  for expected, actual in zip(native.groups, shared.groups, strict=True):
    _assert_state(expected.data, actual.data)
  shared.groups[0].model.opt.run_collision_detection = False
  with pytest.raises(ValueError, match="run_collision_detection"):
    shared.step()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_native_contact_storage_rejects_data_on_another_device():
  model = _model(1, "pyramidal")
  pool = NativeContactTable(16, device="cpu", memory="default")
  with wp.ScopedDevice("cuda:0"):
    with pytest.raises(ValueError, match="current device"):
      io.make_data(model, naconmax=16, contact_storage=pool)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_heterogeneous_native_pool_captured_callback_reads_each_group():
  models = [_model(1, "pyramidal"), _model(6, "pyramidal", objects=2)]
  options = {index: {"naconmax": 16, "njmax": 40} for index in range(2)}
  with wp.ScopedDevice("cuda:0"):
    pool = NativeContactTable(16, device="cuda:0", memory="virtual")
    shared = HeterogeneousWorlds(models, [0, 0, 1], device="cuda:0", data_kwargs=options, contact_storage=pool)
    native = HeterogeneousWorlds(models, [0, 0, 1], device="cuda:0", data_kwargs=options)
    counts = wp.zeros(2, dtype=int)

    def after_group(worlds, group):
      wp.launch(_record_count, dim=1, inputs=[group.data.nacon, group.model_id], outputs=[counts])

    shared.step()
    native.step()
    shared.capture(after_group=after_group)
    for _ in range(3):
      shared.step()
      native.step()
      np.testing.assert_array_equal(counts.numpy(), [int(group.data.nacon.numpy()[0]) for group in native.groups])
      for expected, actual in zip(native.groups, shared.groups, strict=True):
        _assert_state(expected.data, actual.data)
    pool.check_errors()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_shared_native_contacts_replay_all_groups_in_one_graph():
  with wp.ScopedDevice("cuda:0"):
    pool = NativeContactTable(16, device="cuda:0", memory="virtual")
    groups = _groups(pool, "elliptic")
    counts = wp.zeros(len(groups), dtype=int)
    # Warm every model's kernels before capture, preserving identical starting states.
    for model, native, shared in groups:
      forward.step(model, native)
      forward.step(model, shared)
    pointers = [getattr(groups[0][2].contact, field.name).ptr for field in dataclasses.fields(groups[0][2].contact)]
    with wp.ScopedCapture() as capture:
      for index, (model, _, shared) in enumerate(groups):
        forward.step(model, shared)
        pool.observe(shared)
        wp.launch(_record_count, dim=1, inputs=[shared.nacon, index], outputs=[counts])
    for _ in range(4):
      wp.capture_launch(capture.graph)
      expected_counts = []
      for model, native, shared in groups:
        forward.step(model, native)
        expected_counts.append(int(native.nacon.numpy()[0]))
        _assert_state(native, shared)
      np.testing.assert_array_equal(counts.numpy(), expected_counts)
    pool.check_errors()
    assert pointers == [getattr(groups[0][2].contact, field.name).ptr for field in dataclasses.fields(groups[0][2].contact)]

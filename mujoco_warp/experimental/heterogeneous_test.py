"""Behavior tests for compact heterogeneous physics batches."""

import mujoco
import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental import HeterogeneousWorlds
from mujoco_warp.experimental import allocated_bytes


def _models():
  return [
    mujoco.MjModel.from_xml_string("""
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody><body pos="0 0 1"><joint name="joint" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.05"/></body></worldbody>
        <actuator><motor joint="joint"/></actuator>
      </mujoco>"""),
    mujoco.MjModel.from_xml_string("""
      <mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
        <worldbody><body pos="0 0 1"><joint name="first" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/>
          <body pos="0.3 0 0"><joint name="second" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/>
          </body></body></worldbody><actuator><motor joint="first"/><motor joint="second"/></actuator>
      </mujoco>"""),
  ]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compact_mapping_and_zero_copy_world_views(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  worlds = HeterogeneousWorlds(_models(), [1, 0, 1, 0, 1], device=device)
  assert [group.data.qpos.shape for group in worlds.groups] == [(2, 1), (3, 2)]
  np.testing.assert_array_equal(worlds.world_group.numpy(), [1, 0, 1, 0, 1])
  np.testing.assert_array_equal(worlds.world_row.numpy(), [0, 0, 1, 1, 2])
  worlds.world_array(2, "ctrl").fill_(3.0)
  group, row = worlds.locate(2)
  assert row == 1 and group.model_id == 1
  assert worlds.world_array(2, "ctrl").ptr == group.data.ctrl.ptr + group.data.ctrl.strides[0]
  np.testing.assert_array_equal(group.data.ctrl.numpy(), [[0, 0], [3, 3], [0, 0]])
  arrays = worlds.arrays("qpos")
  assert arrays[1][1] is group.data.qpos
  np.testing.assert_array_equal(arrays[1][0].numpy(), [0, 2, 4])
  with pytest.raises(ValueError, match="not indexed by world"):
    worlds.world_array(0, "contact.dist")
  with pytest.raises(IndexError):
    worlds.locate(-1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_different_models_match_independent_batches_and_mujoco(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  models = _models()
  worlds = HeterogeneousWorlds(models, [1, 0, 1], device=device)
  reference = []
  with wp.ScopedDevice(device):
    for group in worlds.groups:
      reference.append((io.put_model(models[group.model_id]), io.make_data(models[group.model_id], group.data.nworld)))
  for group in worlds.groups:
    group.data.ctrl.fill_(0.1)
  for _, data in reference:
    data.ctrl.fill_(0.1)
  for _ in range(5):
    worlds.step()
    with wp.ScopedDevice(device):
      for model, data in reference:
        forward.step(model, data)
  for group, (_, data) in zip(worlds.groups, reference):
    np.testing.assert_allclose(group.data.qpos.numpy(), data.qpos.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(group.data.qvel.numpy(), data.qvel.numpy(), rtol=1e-6, atol=1e-7)
    host = mujoco.MjData(models[group.model_id])
    host.ctrl[:] = 0.1
    for _ in range(5):
      mujoco.mj_step(models[group.model_id], host)
    np.testing.assert_allclose(group.data.qpos.numpy()[0], host.qpos, rtol=1e-4, atol=1e-6)
  report = worlds.memory_report()
  assert report["total_bytes"] == (
    sum(allocated_bytes(group.model) + allocated_bytes(group.data) for group in worlds.groups) + report["mapping_bytes"]
  )


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_one_graph_preserves_capture_state_and_observes_in_place_actions():
  worlds = HeterogeneousWorlds(_models(), [1, 0, 1], device="cuda:0")
  eager = HeterogeneousWorlds(_models(), [1, 0, 1], device="cuda:0")
  worlds.step()
  eager.step()
  before = [(group.data.qpos.numpy(), group.data.qvel.numpy(), group.data.time.numpy()) for group in worlds.groups]
  worlds.capture(steps_per_launch=2)
  for group, (qpos, qvel, time) in zip(worlds.groups, before):
    np.testing.assert_array_equal(group.data.qpos.numpy(), qpos)
    np.testing.assert_array_equal(group.data.qvel.numpy(), qvel)
    np.testing.assert_array_equal(group.data.time.numpy(), time)
  worlds.world_array(2, "ctrl").fill_(0.7)
  eager.world_array(2, "ctrl").fill_(0.7)
  worlds.step()
  eager.step()
  eager.step()
  for group, reference in zip(worlds.groups, eager.groups):
    np.testing.assert_allclose(group.data.qpos.numpy(), reference.data.qpos.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(group.data.time.numpy(), reference.data.time.numpy())


def test_reject_invalid_world_assignment_and_cpu_capture():
  models = _models()
  for ids in ([], [2], [-1], [0.5], [[0]]):
    with pytest.raises(ValueError):
      HeterogeneousWorlds(models, ids, device="cpu")
  worlds = HeterogeneousWorlds(models, [0], device="cpu")
  assert len(worlds.groups) == 1
  with pytest.raises(RuntimeError, match="CUDA"):
    worlds.capture()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_contacts_remain_local_to_each_model_batch():
  models = [
    mujoco.MjModel.from_xml_string(f"""
      <mujoco><option timestep="0.002"/>
        <worldbody><geom type="plane" size="2 2 0.1"/>
          <body pos="0 0 0.09"><freejoint/><geom type="{shape}" size="0.1 0.1 0.1" mass="1"/></body>
        </worldbody>
      </mujoco>""")
    for shape in ("sphere", "box")
  ]
  worlds = HeterogeneousWorlds(models, [1, 0, 1], device="cuda:0")
  reference = HeterogeneousWorlds(models, [1, 0, 1], device="cuda:0")
  worlds.capture()
  for _ in range(3):
    worlds.step()
    reference.step()
  for group, other in zip(worlds.groups, reference.groups):
    np.testing.assert_allclose(group.data.qpos.numpy(), other.data.qpos.numpy(), rtol=1e-5, atol=1e-7)
    assert group.data.nacon.numpy()[0] > 0
    ncontact = int(group.data.nacon.numpy()[0])
    assert np.all(group.data.contact.worldid.numpy()[:ncontact] < group.data.nworld)


@pytest.mark.skipif(not wp.is_cuda_available() or not hasattr(wp, "VirtualMemory"), reason="Warp VMM required")
def test_virtual_data_arena_can_commit_more_pages_and_replay_graph():
  models = _models()
  worlds = HeterogeneousWorlds(models, [1, 0, 1], device="cuda:0", memory="virtual", virtual_reserve_bytes=16 << 20)
  eager = HeterogeneousWorlds(models, [1, 0, 1], device="cuda:0")
  pointers = [group.data.qpos.ptr for group in worlds.groups]
  worlds.capture()
  committed_before = sum(group["virtual_committed_bytes"] for group in worlds.memory_report()["groups"])
  for group in worlds.groups:
    storage = group.virtual_memory["data"]
    storage.commit(storage.reserved_size)
    group.data.ctrl.fill_(0.1)
  for group in eager.groups:
    group.data.ctrl.fill_(0.1)
  worlds.step()
  eager.step()
  for group, reference, pointer in zip(worlds.groups, eager.groups, pointers):
    assert group.data.qpos.ptr == pointer
    storage = group.virtual_memory["data"]
    for field in ("qpos", "qvel", "ctrl", "xpos", "efc.J"):
      parent = group.data
      for name in field.split("."):
        parent = getattr(parent, name)
      if parent.ptr:
        assert storage.ptr <= parent.ptr < storage.ptr + storage.committed_size
    np.testing.assert_allclose(group.data.qpos.numpy(), reference.data.qpos.numpy(), rtol=1e-6, atol=1e-7)
  assert all(
    group["virtual_reserved_bytes"] >= group["virtual_committed_bytes"] > 0 for group in worlds.memory_report()["groups"]
  )
  assert sum(group["virtual_committed_bytes"] for group in worlds.memory_report()["groups"]) > committed_before

"""Contact snapshots preserve native rows and detect incomplete exports."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp.experimental.contact_table import ContactTable


def _contacts(device, native_capacity=16):
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
      <body pos="1 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>
  """)
  with wp.ScopedDevice(device):
    wm = io.put_model(model)
    data = io.make_data(model, nworld=3, naconmax=native_capacity)
    qpos = np.tile(model.qpos0, (3, 1))
    qpos[2, 9] = 0.5
    wp.copy(data.qpos, wp.array(qpos, dtype=float, device=device))
    mapping = wp.array([2, -1, 0], dtype=int, device=device)
  return wm, SimpleNamespace(data=data, world_ids=mapping)


def _assert_native_rows(snapshot, groups):
  expected = [[] for _ in range(snapshot.table.state.num_worlds)]
  for group in groups:
    count = int(group.data.nacon.numpy()[0])
    assert count <= group.data.naconmax
    contact = group.data.contact
    mapping = group.world_ids.numpy()
    physical = contact.worldid.numpy()[:count]
    position = contact.pos.numpy()[:count]
    normal = contact.frame.numpy()[:count, 0]
    depth = -contact.dist.numpy()[:count]
    geom = contact.geom.numpy()[:count]
    for source, slot in enumerate(physical):
      world = mapping[slot]
      if world >= 0:
        expected[world].append(np.r_[position[source], normal[source], depth[source], geom[source]])
  offsets = snapshot.world_offsets.numpy()
  np.testing.assert_array_equal(np.diff(offsets), [len(rows) for rows in expected])
  columns = snapshot.columns
  actual = np.column_stack([columns[name].numpy()[: offsets[-1]] for name in columns])
  for world, rows in enumerate(expected):
    observed = actual[offsets[world] : offsets[world + 1]]
    # Compaction and parallel appends deliberately leave row order unspecified.
    observed = sorted(observed.tolist())
    np.testing.assert_allclose(observed, sorted(np.asarray(rows).tolist()), atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_snapshot_matches_native_contacts_and_replays_after_growth(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA required")
  wm, group = _contacts(device)
  snapshot = ContactTable(3, max_rows=16, initial_capacity=1, device=device, dispatch_width=8)

  def rebuild():
    with wp.ScopedDevice(device):
      forward.forward(wm, group.data)
      snapshot.rebuild([group])

  rebuild()  # Warm kernels and persistent scan scratch before CUDA capture.
  graph = None
  if device.startswith("cuda"):
    with wp.ScopedCapture(device=device) as capture:
      rebuild()
    graph = capture.graph
    wp.capture_launch(graph)
  with pytest.raises(RuntimeError, match="committed row capacity exhausted"):
    snapshot.check_errors()
  pointer = snapshot.columns["position"].ptr
  snapshot.table.ensure_capacity(8)
  snapshot.clear_errors()
  if graph is None:
    rebuild()
  else:
    wp.capture_launch(graph)
  snapshot.check_errors()
  _assert_native_rows(snapshot, [group])
  np.testing.assert_array_equal(np.diff(snapshot.world_offsets.numpy()), [1, 0, 2])

  # Change native contacts and logical mapping, using the same graph and columns.
  qpos = group.data.qpos.numpy()
  qpos[0, [2, 9]] = 0.5
  qpos[1, [2, 9]] = 0.09
  wp.copy(group.data.qpos, wp.array(qpos, dtype=float, device=device))
  wp.copy(group.world_ids, wp.array([2, 1, 0], dtype=int, device=device))
  if graph is None:
    rebuild()
  else:
    wp.capture_launch(graph)
  snapshot.check_errors()
  _assert_native_rows(snapshot, [group])
  np.testing.assert_array_equal(np.diff(snapshot.world_offsets.numpy()), [1, 2, 0])
  assert snapshot.columns["position"].ptr == pointer


def test_snapshot_reports_native_contact_overflow():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>
  """)
  with wp.ScopedDevice("cpu"):
    data = io.make_data(model, naconmax=1)
    forward.forward(io.put_model(model), data)
    group = SimpleNamespace(data=data, world_ids=wp.array([0], dtype=int))
  assert data.nacon.numpy()[0] > data.naconmax
  snapshot = ContactTable(1, max_rows=8, initial_capacity=8, device="cpu", dispatch_width=8)
  snapshot.rebuild([group])
  with pytest.raises(RuntimeError, match="Native MJWarp contact/pair capacity exhausted"):
    snapshot.check_errors()
  assert snapshot.table.live_count <= group.data.naconmax


def test_snapshot_reports_native_broadphase_overflow_before_contact_overflow():
  wm, group = _contacts("cpu", native_capacity=1)
  with wp.ScopedDevice("cpu"):
    forward.forward(wm, group.data)
  assert group.data.ncollision.numpy()[0] > group.data.naconmax
  assert group.data.nacon.numpy()[0] <= group.data.naconmax
  snapshot = ContactTable(3, max_rows=8, initial_capacity=8, device="cpu", dispatch_width=8)
  snapshot.rebuild([group])
  with pytest.raises(RuntimeError, match="Native MJWarp contact/pair capacity exhausted"):
    snapshot.check_errors()


def test_snapshot_bulk_reservation_combines_model_groups():
  wm, first = _contacts("cpu")
  _, second = _contacts("cpu")
  wp.copy(second.world_ids, wp.array([-1, 1, -1], dtype=int, device="cpu"))
  with wp.ScopedDevice("cpu"):
    forward.forward(wm, first.data)
    forward.forward(wm, second.data)
  snapshot = ContactTable(3, max_rows=16, initial_capacity=8, device="cpu", dispatch_width=8)
  snapshot.rebuild([first, second])
  snapshot.check_errors()
  _assert_native_rows(snapshot, [first, second])
  np.testing.assert_array_equal(np.diff(snapshot.world_offsets.numpy()), [1, 2, 2])

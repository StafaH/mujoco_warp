"""Record real heterogeneous mjwarp collisions and compare contact capacity budgets.

uv run --no-sync python -m mujoco_warp.experimental.dynamic_contacts_example \
  --worlds 8192 --output mujoco_warp/experimental/dynamic_contacts_demo.html

GPU kernels reset falling objects and rebuild a downstream DynamicTable inside one
unchanged CUDA graph. Native mjwarp contacts still drive the solver. HTML playback
uses GPU-recorded positions and contacts; the browser does not simulate physics.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp.experimental import HeterogeneousWorlds
from mujoco_warp.experimental import allocated_bytes
from mujoco_warp.experimental.contact_table import ContactTable

wp.set_module_options({"enable_backward": False})

OBJECTS = (1, 4, 12)
STEPS = 384
SAMPLE = 4
MAX_VISIBLE_CONTACTS = 96


@wp.kernel
def _reset(
  # In:
  tick: wp.array[int],
  world_ids: wp.array[int],
  defaults: wp.array[float],
  # Data out:
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
  qacc_warmstart_out: wp.array2d[float],
):
  row = wp.tid()
  world = world_ids[row]
  if tick[0] == 0 or (tick[0] + world % 3 * 16) % 128 == 0:
    for index in range(qpos_out.shape[1]):
      qpos_out[row, index] = defaults[index]
    for index in range(qvel_out.shape[1]):
      qvel_out[row, index] = 0.0
      qacc_warmstart_out[row, index] = 0.0


@wp.kernel
def _record_objects(
  # In:
  tick: wp.array[int],
  visible: wp.array[int],
  world_group: wp.array[int],
  world_row: wp.array[int],
  group_id: int,
  object_count: int,
  positions: wp.array2d[wp.vec3],
  # Out:
  position_history_out: wp.array3d[wp.vec3],
):
  view, obj = wp.tid()
  world = visible[view]
  if tick[0] % SAMPLE == 0 and world_group[world] == group_id and obj < object_count:
    position_history_out[tick[0] // SAMPLE, view, obj] = positions[world_row[world], obj + 1]


@wp.kernel
def _record_table(
  # In:
  tick: wp.array[int],
  visible: wp.array[int],
  offsets: wp.array[int],
  position: wp.array[wp.vec3],
  normal: wp.array[wp.vec3],
  depth: wp.array[float],
  num_worlds: int,
  # Out:
  counts_out: wp.array2d[int],
  point_history_out: wp.array3d[wp.vec3],
  normal_history_out: wp.array3d[wp.vec3],
  depth_history_out: wp.array3d[float],
  totals_out: wp.array[int],
  errors_out: wp.array[int],
):
  view = wp.tid()
  if tick[0] % SAMPLE == 0:
    frame = tick[0] // SAMPLE
    world = visible[view]
    count = offsets[world + 1] - offsets[world]
    counts_out[frame, view] = count
    if count > MAX_VISIBLE_CONTACTS:
      wp.atomic_or(errors_out, 0, 1)
    for local in range(wp.min(count, MAX_VISIBLE_CONTACTS)):
      row = offsets[world] + local
      point_history_out[frame, view, local] = position[row]
      normal_history_out[frame, view, local] = normal[row]
      depth_history_out[frame, view, local] = depth[row]
    if view == 0:
      totals_out[frame] = offsets[num_worlds]


@wp.kernel
def _validate_native(
  # In:
  contact_count: wp.array[int],
  pair_count: wp.array[int],
  capacity: int,
  constraint_count: wp.array[int],
  constraint_capacity: int,
  # Out:
  errors_out: wp.array[int],
):
  row = wp.tid()
  if contact_count[0] > capacity or pair_count[0] > capacity:
    wp.atomic_or(errors_out, 0, 2)
  if constraint_count[row] > constraint_capacity:
    wp.atomic_or(errors_out, 0, 4)


@wp.kernel
def _advance(
  # Out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


def _model(objects):
  bodies = "".join(
    f'<body pos="{0.4 + (i % 4) * 0.42} 0 {0.5 + (i // 4) * 0.25}">'
    '<freejoint/><geom type="sphere" size="0.09" mass="0.1" condim="3"/></body>'
    for i in range(objects)
  )
  return mujoco.MjModel.from_xml_string(
    '<mujoco><option timestep="0.005" jacobian="sparse" iterations="10"/>'
    '<worldbody><geom type="plane" size="3 3 0.1"/>' + bodies + "</worldbody></mujoco>"
  )


def _run(models, model_ids, *, tight, record, snapshot=False):
  nworld = len(model_ids)
  options = {
    index: {"nconmax": 4 * objects if tight else 128, "njmax": 12 * objects + 16} for index, objects in enumerate(OBJECTS)
  }
  worlds = HeterogeneousWorlds(models, model_ids, device="cuda:0", data_kwargs=options)
  device = worlds.device
  tick = wp.zeros(1, dtype=int, device=device)
  errors = wp.zeros(1, dtype=int, device=device)
  defaults = [wp.array(models[group.model_id].qpos0, dtype=float, device=device) for group in worlds.groups]
  visible_ids = list(range(min(nworld, 9)))
  visible = wp.array(visible_ids, dtype=int, device=device)
  frames = STEPS // SAMPLE
  views = len(visible_ids)
  history = {}
  contact_table = None
  if record or snapshot:
    contact_table = ContactTable(
      nworld, max_rows=nworld * 128, initial_capacity=4 * sum(OBJECTS[i] for i in model_ids), device=device
    )
  if record:
    history = {
      "positions": wp.zeros((frames, views, max(OBJECTS)), dtype=wp.vec3, device=device),
      "points": wp.zeros((frames, views, MAX_VISIBLE_CONTACTS), dtype=wp.vec3, device=device),
      "normals": wp.zeros((frames, views, MAX_VISIBLE_CONTACTS), dtype=wp.vec3, device=device),
      "depths": wp.zeros((frames, views, MAX_VISIBLE_CONTACTS), dtype=float, device=device),
      "counts": wp.zeros((frames, views), dtype=int, device=device),
      "totals": wp.zeros(frames, dtype=int, device=device),
    }

  def step():
    for group, default in zip(worlds.groups, defaults, strict=True):
      wp.launch(
        _reset,
        dim=group.data.nworld,
        inputs=[tick, group.world_ids, default, group.data.qpos, group.data.qvel, group.data.qacc_warmstart],
        device=device,
      )
      forward.step(group.model, group.data)
      wp.launch(
        _validate_native,
        dim=group.data.nworld,
        inputs=[group.data.nacon, group.data.ncollision, group.data.naconmax, group.data.nefc, group.data.njmax, errors],
        device=device,
      )
      if record:
        wp.launch(
          _record_objects,
          dim=(views, max(OBJECTS)),
          inputs=[
            tick,
            visible,
            worlds.world_group,
            worlds.world_row,
            group.model_id,
            OBJECTS[group.model_id],
            group.data.geom_xpos,
            history["positions"],
          ],
          device=device,
        )
    if contact_table is not None:
      contact_table.rebuild(worlds.groups)
    if record:
      wp.launch(
        _record_table,
        dim=views,
        inputs=[
          tick,
          visible,
          contact_table.world_offsets,
          contact_table.columns["position"],
          contact_table.columns["normal"],
          contact_table.columns["depth"],
          nworld,
          history["counts"],
          history["points"],
          history["normals"],
          history["depths"],
          history["totals"],
          errors,
        ],
        device=device,
      )
    wp.launch(_advance, dim=1, inputs=[tick], device=device)

  # Compile all kernels before capture. Tick zero resets this warm-up's physical state.
  step()
  tick.zero_()
  errors.zero_()
  if contact_table is not None:
    contact_table.clear_errors()
  with wp.ScopedCapture(device=device) as capture:
    step()
  table_pointer = contact_table.columns["position"].ptr if contact_table is not None else None
  table_bytes = contact_table.table.committed_bytes + contact_table.workspace_bytes if contact_table is not None else 0
  start = wp.Event(device=device, enable_timing=True)
  end = wp.Event(device=device, enable_timing=True)
  wp.record_event(start)
  for _ in range(STEPS):
    wp.capture_launch(capture.graph)
  wp.record_event(end)
  wp.synchronize_device(device)
  if errors.numpy()[0] != 0:
    raise RuntimeError(f"Native contact/constraint or recording overflow: {errors.numpy()[0]}")
  if contact_table is not None:
    contact_table.check_errors()
    if (
      contact_table.columns["position"].ptr != table_pointer
      or contact_table.table.committed_bytes + contact_table.workspace_bytes != table_bytes
    ):
      raise RuntimeError("Contact table storage changed during replay")
  qpos = [group.data.qpos.numpy() for group in worlds.groups]
  if not all(np.all(np.isfinite(value)) for value in qpos):
    raise RuntimeError("Nonfinite physics state")
  metrics = {
    "physics_bytes": worlds.memory_report()["total_bytes"],
    "contact_bytes": sum(allocated_bytes(group.data.contact) for group in worlds.groups),
    "capacity": sum(group.data.naconmax for group in worlds.groups),
    "rollout_ms": wp.get_event_elapsed_time(start, end),
  }
  recording_bytes = allocated_bytes(list(history.values()))
  arrays = {name: array.numpy() for name, array in history.items()}
  return metrics, qpos, arrays, visible_ids, table_bytes, recording_bytes


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worlds", type=int, default=8192)
  parser.add_argument("--output", type=Path, default=Path("dynamic_contacts_demo.html"))
  args = parser.parse_args()
  if args.worlds < 1:
    parser.error("--worlds must be positive")
  wp.init()
  models = [_model(objects) for objects in OBJECTS]
  ids = np.arange(args.worlds, dtype=np.int32) % len(OBJECTS)
  generous, reference, _, _, _, _ = _run(models, ids, tight=False, record=False)
  tight, tight_state, _, _, _, _ = _run(models, ids, tight=True, record=False)
  snapshot, actual, arrays, visible, table_bytes, recording_bytes = _run(models, ids, tight=True, record=True)
  max_difference = 0.0
  for expected, native, observed in zip(reference, tight_state, actual, strict=True):
    np.testing.assert_allclose(native, expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(observed, expected, atol=1e-4, rtol=1e-4)
    max_difference = max(max_difference, float(np.max(np.abs(expected - observed))))
  if not np.any(arrays["totals"] > 0) or not np.any(arrays["totals"] == 0):
    raise RuntimeError("Recording must include both contact-free and contacting frames")
  frames = []
  for frame in range(STEPS // SAMPLE):
    world_frames = []
    for view, world in enumerate(visible):
      count = int(arrays["counts"][frame, view])
      contacts = []
      for local in range(count):
        point = arrays["points"][frame, view, local]
        normal = arrays["normals"][frame, view, local]
        contacts.append(
          [float(point[0]), float(point[2]), float(normal[0]), float(normal[2]), float(arrays["depths"][frame, view, local])]
        )
      world_frames.append(
        {
          "id": world,
          "particles": arrays["positions"][frame, view, : OBJECTS[ids[world]], :][:, [0, 2]].tolist(),
          "contacts": contacts,
        }
      )
    frames.append(
      {
        "phase": f"Drop cycle {frame * SAMPLE // 128 + 1}",
        "total_particles": int(sum(OBJECTS[i] for i in ids)),
        "total_contacts": int(arrays["totals"][frame]),
        "worlds": world_frames,
      }
    )
  payload = {
    "worlds": args.worlds,
    "steps": STEPS,
    "hardware": wp.get_device("cuda:0").name,
    "visible_worlds": visible,
    "frames": frames,
    "comparison": {
      "generous": generous,
      "tight": tight,
      "table_bytes": table_bytes,
      "table_recording_rollout_ms": snapshot["rollout_ms"],
      "recording_bytes": recording_bytes,
      "validated": True,
      "max_position_difference": max_difference,
    },
    "notes": [
      "Real mjwarp collision detection and constraint solving; worlds contain 1, 4, or 12 free spheres and a ground plane.",
      "GPU kernels reset drops every 128 steps, staggered by world. No host reset, allocation, readback, or recapture during replay.",
      "Objects and model assignments are fixed in this example. Contact rows appear and disappear; they are rebuilt each step.",
      "Native mjwarp already packs contacts globally per model batch. The reduction uses a tighter nconmax/naconmax budget.",
      "DynamicTable copies contact observations after physics for GPU consumers and export. It adds memory; it does not replace solver storage.",
      "Committed table memory stays fixed during graph replay. Physical growth still requires a host boundary outside capture.",
      "Array byte totals exclude CUDA context, graphs, compiled kernels, allocator cache, scan utility workspace, and tiny reset/validation buffers.",
      "Timing is one rollout, not a throughput benchmark. Native runs do equal work; the separately timed snapshot run adds table rebuilding and GPU recording.",
    ],
  }
  template = Path(__file__).with_name("assets").joinpath("dynamic_contacts.html").read_text()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(template.replace("__DEMO_DATA__", json.dumps(payload)))
  args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2))
  print(
    json.dumps(
      {
        "output": str(args.output.resolve()),
        "comparison": payload["comparison"],
        "contact_range": [int(arrays["totals"].min()), int(arrays["totals"].max())],
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()

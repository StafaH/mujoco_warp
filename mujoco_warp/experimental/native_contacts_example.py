"""Use shared contact columns directly in native collision detection and the solver.

uv run --no-sync python -m mujoco_warp.experimental.native_contacts_example \
  --worlds 8192 --repeats 7 --output mujoco_warp/experimental/native_contacts_demo.html

Complete model steps run sequentially against one shared native contact table. There
is no DynamicTable snapshot, contact payload copy, or ECS compaction. GPU resets and
optional visual recording remain inside one graph; inspection happens after rollout.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import warp as wp

from mujoco_warp._src import forward
from mujoco_warp.experimental import HeterogeneousWorlds
from mujoco_warp.experimental import NativeContactTable
from mujoco_warp.experimental import allocated_bytes
from mujoco_warp.experimental.dynamic_contacts_example import MAX_VISIBLE_CONTACTS
from mujoco_warp.experimental.dynamic_contacts_example import OBJECTS
from mujoco_warp.experimental.dynamic_contacts_example import SAMPLE
from mujoco_warp.experimental.dynamic_contacts_example import STEPS
from mujoco_warp.experimental.dynamic_contacts_example import _advance
from mujoco_warp.experimental.dynamic_contacts_example import _model
from mujoco_warp.experimental.dynamic_contacts_example import _record_objects
from mujoco_warp.experimental.dynamic_contacts_example import _reset
from mujoco_warp.experimental.dynamic_contacts_example import _validate_native

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _record_native_contacts(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  tick: wp.array[int],
  world_ids: wp.array[int],
  visible_index: wp.array[int],
  native_world: wp.array[int],
  position: wp.array[wp.vec3],
  frame: wp.array[wp.mat33],
  distance: wp.array[float],
  capacity: int,
  width: int,
  # Out:
  recorded_counts_out: wp.array2d[int],
  point_history_out: wp.array3d[wp.vec3],
  normal_history_out: wp.array3d[wp.vec3],
  depth_history_out: wp.array3d[float],
  totals_out: wp.array[int],
  errors_out: wp.array[int],
):
  if tick[0] % SAMPLE != 0:
    return
  sample = tick[0] // SAMPLE
  if wp.tid() == 0:
    wp.atomic_add(totals_out, sample, nacon_in[0])
  for contact in range(wp.tid(), wp.min(nacon_in[0], capacity), width):
    world = world_ids[native_world[contact]]
    if world >= 0:
      view = visible_index[world]
      if view >= 0:
        local = wp.atomic_add(recorded_counts_out, sample, view, 1)
        if local < MAX_VISIBLE_CONTACTS:
          point_history_out[sample, view, local] = position[contact]
          normal_history_out[sample, view, local] = frame[contact][0]
          depth_history_out[sample, view, local] = -distance[contact]
        else:
          wp.atomic_or(errors_out, 0, 1)


def run_native(models, ids, *, shared=False, memory="virtual", record=False):
  """Run equal native physics with independent or directly shared contact storage."""
  device = wp.get_device("cuda:0")
  sizes = np.bincount(ids, minlength=len(models))
  budgets = [int(sizes[i]) * 4 * objects for i, objects in enumerate(OBJECTS)]
  width = max(max(1, 2 * (int(np.max(model.geom_condim)) - 1)) for model in models)
  pool = NativeContactTable(max(budgets), max_pyramid=width, device=device, memory=memory) if shared else None
  options = {i: {"nconmax": 4 * objects, "njmax": 12 * objects + 16} for i, objects in enumerate(OBJECTS)}
  worlds = HeterogeneousWorlds(models, ids, device=device, data_kwargs=options, contact_storage=pool)
  tick = wp.zeros(1, dtype=int, device=device)
  errors = wp.zeros(1, dtype=int, device=device)
  defaults = [wp.array(models[group.model_id].qpos0, dtype=float, device=device) for group in worlds.groups]
  visible_ids = list(range(min(len(ids), 9)))
  visible = wp.array(visible_ids, dtype=int, device=device)
  index = np.full(len(ids), -1, dtype=np.int32)
  index[visible_ids] = np.arange(len(visible_ids), dtype=np.int32)
  visible_index = wp.array(index, dtype=int, device=device)
  frames, views = STEPS // SAMPLE, len(visible_ids)
  history = {}
  if record:
    history = {
      "positions": wp.zeros((frames, views, max(OBJECTS)), dtype=wp.vec3, device=device),
      "points": wp.zeros((frames, views, MAX_VISIBLE_CONTACTS), dtype=wp.vec3, device=device),
      "normals": wp.zeros((frames, views, MAX_VISIBLE_CONTACTS), dtype=wp.vec3, device=device),
      "depths": wp.zeros((frames, views, MAX_VISIBLE_CONTACTS), dtype=float, device=device),
      "counts": wp.zeros((frames, views), dtype=int, device=device),
      "totals": wp.zeros(frames, dtype=int, device=device),
    }

  def before_step(batch):
    for group, default in zip(batch.groups, defaults, strict=True):
      wp.launch(
        _reset,
        dim=group.data.nworld,
        inputs=[tick, group.world_ids, default, group.data.qpos, group.data.qvel, group.data.qacc_warmstart],
        device=device,
      )

  def after_group(batch, group):
    data = group.data
    wp.launch(
      _validate_native,
      dim=data.nworld,
      inputs=[data.nacon, data.ncollision, data.naconmax, data.nefc, data.njmax, errors],
      device=device,
    )
    if record:
      wp.launch(
        _record_objects,
        dim=(views, max(OBJECTS)),
        inputs=[
          tick,
          visible,
          batch.world_group,
          batch.world_row,
          group.model_id,
          OBJECTS[group.model_id],
          data.geom_xpos,
          history["positions"],
        ],
        device=device,
      )
      wp.launch(
        _record_native_contacts,
        dim=4096,
        inputs=[
          data.nacon,
          tick,
          group.world_ids,
          visible_index,
          data.contact.worldid,
          data.contact.pos,
          data.contact.frame,
          data.contact.dist,
          data.naconmax,
          4096,
        ],
        outputs=[history["counts"], history["points"], history["normals"], history["depths"], history["totals"], errors],
        device=device,
      )
    if group is batch.groups[-1]:
      wp.launch(_advance, dim=1, inputs=[tick], device=device)

  # Compile everything without advancing the recorded episode's tick.
  before_step(worlds)
  for group in worlds.groups:
    forward.step(group.model, group.data)
    if pool is not None:
      pool.observe(group.data)
    after_group(worlds, group)
  tick.zero_()
  errors.zero_()
  for array in history.values():
    array.zero_()
  if pool is not None:
    pool.clear_errors()
    for group in worlds.groups:
      for name, column in pool.columns.items():
        if getattr(group.data.contact, name).ptr != column.ptr:
          raise RuntimeError("A native contact component does not alias shared storage")
  worlds.capture(before_step=before_step, after_group=after_group)
  graph = worlds.graph
  pointers = [group.data.contact.pos.ptr for group in worlds.groups]
  start, end = wp.Event(device=device, enable_timing=True), wp.Event(device=device, enable_timing=True)
  wp.record_event(start)
  for _ in range(STEPS):
    worlds.step()
  wp.record_event(end)
  wp.synchronize_device(device)
  if errors.numpy()[0]:
    raise RuntimeError(f"Native contact/constraint/recording overflow: {errors.numpy()[0]}")
  if pool is not None:
    pool.check_errors()
  if worlds.graph is not graph or pointers != [group.data.contact.pos.ptr for group in worlds.groups]:
    raise RuntimeError("Graph or contact addresses changed during replay")
  state = [(group.data.qpos.numpy(), group.data.qvel.numpy()) for group in worlds.groups]
  if not all(np.all(np.isfinite(array)) for pair in state for array in pair):
    raise RuntimeError("Nonfinite simulation state")
  report = worlds.memory_report()
  metrics = {
    "physics_bytes": report["total_bytes"],
    "contact_bytes": pool.allocated_bytes if pool else sum(allocated_bytes(g.data.contact) for g in worlds.groups),
    "capacity": max(budgets) if pool else sum(budgets),
    "per_group_capacities": [g.data.naconmax for g in worlds.groups],
    "rollout_ms": wp.get_event_elapsed_time(start, end),
    "recording_bytes": allocated_bytes(list(history.values())),
  }
  arrays = {name: array.numpy() for name, array in history.items()}
  return metrics, state, arrays, visible_ids


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worlds", type=int, default=8192)
  parser.add_argument("--repeats", type=int, default=7)
  parser.add_argument("--output", type=Path, default=Path("native_contacts_demo.html"))
  args = parser.parse_args()
  if args.worlds < 1 or args.repeats < 1:
    parser.error("--worlds and --repeats must be positive")
  wp.init()
  models = [_model(count) for count in OBJECTS]
  ids = np.arange(args.worlds, dtype=np.int32) % len(models)
  samples = {name: [] for name in ("independent", "shared_default", "shared_virtual")}
  metrics = {}
  difference = 0.0
  for trial in range(args.repeats):
    configs = [("independent", False, "default"), ("shared_default", True, "default"), ("shared_virtual", True, "virtual")]
    reference = None
    for name, shared, memory in configs if trial % 2 == 0 else configs[::-1]:
      result, state, _, _ = run_native(models, ids, shared=shared, memory=memory)
      metrics[name] = result
      samples[name].append(result["rollout_ms"])
      if reference is None:
        reference = state
      else:
        for expected, observed in zip(reference, state, strict=True):
          for a, b in zip(expected, observed, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
            difference = max(difference, float(np.max(np.abs(a - b))))
      gc.collect()
    print(f"Trial {trial + 1}/{args.repeats} complete", flush=True)
  _, reference, baseline_frames, visible = run_native(models, ids, record=True)
  recording, observed, arrays, _ = run_native(models, ids, shared=True, record=True)
  for first, second in zip(reference, observed, strict=True):
    for a, b in zip(first, second, strict=True):
      np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
  np.testing.assert_array_equal(arrays["counts"], baseline_frames["counts"])
  np.testing.assert_array_equal(arrays["totals"], baseline_frames["totals"])
  np.testing.assert_allclose(arrays["positions"], baseline_frames["positions"], atol=1e-4, rtol=1e-4)
  if not np.any(arrays["totals"] > 0) or not np.any(arrays["totals"] == 0):
    raise RuntimeError("Recording must show contacts appearing and disappearing")
  frames = []
  for frame in range(STEPS // SAMPLE):
    worlds = []
    for view, world in enumerate(visible):
      count = int(arrays["counts"][frame, view])
      contacts = [
        [
          float(arrays["points"][frame, view, local, 0]),
          float(arrays["points"][frame, view, local, 2]),
          float(arrays["normals"][frame, view, local, 0]),
          float(arrays["normals"][frame, view, local, 2]),
          float(arrays["depths"][frame, view, local]),
        ]
        for local in range(count)
      ]
      worlds.append(
        {
          "id": world,
          "particles": arrays["positions"][frame, view, : OBJECTS[ids[world]]][:, [0, 2]].tolist(),
          "contacts": contacts,
        }
      )
    frames.append(
      {
        "phase": f"Drop cycle {frame * SAMPLE // 128 + 1}",
        "total_particles": int(sum(OBJECTS[i] for i in ids)),
        "total_contacts": int(arrays["totals"][frame]),
        "worlds": worlds,
      }
    )
  for name in metrics:
    metrics[name]["rollout_ms"] = float(np.median(samples[name]))
    metrics[name]["samples_ms"] = samples[name]
  payload = {
    "worlds": args.worlds,
    "steps": STEPS,
    "repeats": args.repeats,
    "hardware": wp.get_device("cuda:0").name,
    "visible_worlds": visible,
    "frames": frames,
    "metrics": metrics,
    "comparison": {
      "generous": metrics["independent"],
      "tight": metrics["shared_virtual"],
      "table_bytes": 0,
      "recording_bytes": recording["recording_bytes"],
      "validated": True,
      "max_position_difference": difference,
    },
    "notes": [
      "Native collision detection writes directly to shared columns consumed by the native solver.",
      "No snapshot table, payload copy, entity handles, or compaction. Native contact row order is preserved.",
      "Per-group contact budgets are unchanged; storage is reused only after each complete model step and its consumers.",
      "GPU reset and recording run under one unchanged graph; all inspection happens after rollout.",
      "Shared contacts are transient. Read them in after_group; they are overwritten by the next model batch.",
      "Capacity is committed at startup. There is no on-device physical growth or host reset.",
      "Seven-trial medians compare equal native work without recording; default and virtual shared storage separate pooling from VMM.",
      "Reported array storage excludes compiled kernels, CUDA graphs/context, allocator cache, collision/scan scratch, and small reset buffers.",
    ],
  }
  template = Path(__file__).with_name("assets").joinpath("native_contacts.html").read_text()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(template.replace("__DEMO_DATA__", json.dumps(payload)))
  args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2))
  print(
    json.dumps(
      {
        "output": str(args.output.resolve()),
        "metrics": metrics,
        "max_state_difference": difference,
        "contact_range": [int(arrays["totals"].min()), int(arrays["totals"].max())],
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()

"""One graph changes all worlds' model populations using a shared physics workspace.

uv run --no-sync benchmarks/heterogeneous_pooled.py --batch-size 4096 \
  --output benchmarks/heterogeneous_results/pooled_4096_8192.json
"""

import argparse
import json
import pathlib

import numpy as np
import warp as wp
from heterogeneous import make_models

from mujoco_warp.experimental.pooled import PooledHeterogeneousWorlds

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _episodes(
  # In:
  tick: wp.array[int],
  current_model: wp.array[int],
  stage_steps: int,
  # Out:
  requests_out: wp.array[int],
  counts_out: wp.array2d[int],
):
  world = wp.tid()
  step = tick[0]
  stage = step // stage_steps
  if step % stage_steps == 0:
    model = world % 3
    if stage == 1:
      model = 1
    elif stage == 2:
      model = 2
    requests_out[world] = model
  elif step % 200 == 0:
    requests_out[world] = current_model[world]
  if step % stage_steps == stage_steps - 1:
    wp.atomic_add(counts_out, stage, current_model[world], 1)


@wp.kernel
def _tick(
  # Data out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--nworld", type=int, default=8192)
  parser.add_argument("--batch-size", type=int, default=4096)
  parser.add_argument("--scratch-mib", type=int, default=64)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--output", type=pathlib.Path)
  args = parser.parse_args()
  if min(args.nworld, args.batch_size, args.scratch_mib, args.repeats) < 1:
    parser.error("All counts must be positive")
  models, _ = make_models()
  worlds = PooledHeterogeneousWorlds(
    models,
    np.arange(args.nworld, dtype=np.int32) % 3,
    device="cuda:0",
    batch_size=args.batch_size,
    scratch_bytes=args.scratch_mib << 20,
    data_kwargs={i: {"nconmax": 16, "njmax": 64} for i in range(3)},
  )
  tick = wp.zeros(1, dtype=int, device=worlds.device)
  counts = wp.zeros((4, 3), dtype=int, device=worlds.device)
  stage_steps = (5 + args.repeats) * 200

  def episodes(batch):
    wp.launch(
      _episodes,
      dim=args.nworld,
      inputs=[tick, batch.world_model_ids, stage_steps],
      outputs=[batch.reset_model_ids, counts],
      device=batch.device,
    )
    wp.launch(_tick, dim=1, outputs=[tick], device=batch.device)

  def controls(batch):
    batch.state.ctrl.fill_(0.05)

  worlds.capture(steps_per_launch=10, before_reset=episodes, before_step=controls)
  graph = worlds.graph
  pointers = (worlds.state.qpos.ptr, worlds.state.qvel.ptr)
  stages = []
  # The host only submits the same graph and collects timing events. Model selection,
  # resets and chunk scheduling use GPU state; no mapping readbacks or CPU reset calls.
  for name in ("balanced", "all_humanoid", "all_single_arm", "balanced_again"):
    durations = []
    for sample in range(5 + args.repeats):
      start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
      wp.record_event(start)
      for _ in range(20):
        worlds.step()
      wp.record_event(end)
      elapsed = wp.get_event_elapsed_time(start, end) / 1000
      if sample >= 5:
        durations.append(elapsed)
    stages.append(
      {
        "name": name,
        "batch_step_ms": float(np.median(durations)) * 1000 / 200,
        "active_environment_steps_per_second": args.nworld * 200 / float(np.median(durations)),
        "sample_seconds": durations,
      }
    )
  assert worlds.graph is graph and (worlds.state.qpos.ptr, worlds.state.qvel.ptr) == pointers
  assert np.isfinite(worlds.state.qpos.numpy()).all()
  assert worlds.device_reset_status.numpy()[0] == 0
  assert not np.any(worlds.physics_overflow.numpy())
  assert not np.any(worlds.physics_nonfinite.numpy())
  histogram = counts.numpy()
  expected_mixed = np.bincount(np.arange(args.nworld) % 3, minlength=3)
  np.testing.assert_array_equal(histogram, [expected_mixed, [0, args.nworld, 0], [0, 0, args.nworld], expected_mixed])
  report = {
    "nworld": args.nworld,
    "batch_size": args.batch_size,
    "memory": worlds.memory_report(),
    "stages": stages,
    "stage_model_counts": histogram.tolist(),
    "committed_world_resets": int(worlds.device_reset_epoch.numpy().sum()),
    "graph_reused": True,
    "persistent_state_pointers_reused": True,
    "device_reset_status": worlds.device_reset_status.numpy().tolist(),
    "physics_overflow_worlds": int(np.count_nonzero(worlds.physics_overflow.numpy())),
    "physics_nonfinite_worlds": int(np.count_nonzero(worlds.physics_nonfinite.numpy())),
    "notes": "All population changes, episode resets and chunk scheduling execute on GPU inside one graph. "
    "Each stage has five warmup samples and the requested timed 200-step samples. "
    "Reported memory includes preallocated capture scratch; older fixed-batch array reports exclude graph scratch.",
  }
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()

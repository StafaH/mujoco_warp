"""GPU-generated model changes with no reset readback, synchronization, or recapture.

uv run --no-sync benchmarks/heterogeneous_device_reset.py --switch-worlds 96 --period 1 \
  --output benchmarks/heterogeneous_results/device_reset_96_8192.json
"""

import argparse
import json
import pathlib

import numpy as np
import warp as wp
from heterogeneous import make_models

from mujoco_warp._src import io
from mujoco_warp.experimental import HeterogeneousWorlds

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _request_switches(
  # In:
  tick: wp.array[int],
  current_models: wp.array[int],
  period: int,
  switch_worlds: int,
  # Out:
  requests_out: wp.array[int],
):
  world = wp.tid()
  if tick[0] % 200 == 0:
    requests_out[world] = current_models[world]
  elif world < switch_worlds and tick[0] % period == 0:
    requests_out[world] = (current_models[world] + 1) % 3


@wp.kernel
def _reset_due(
  # In:
  tick: wp.array[int],
  # Out:
  due_out: wp.array[int],
):
  due_out[0] = int(tick[0] % 200 == 0)


@wp.kernel
def _advance_tick(
  # Data out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--nworld", type=int, default=8192)
  parser.add_argument("--switch-worlds", type=int, default=96)
  parser.add_argument("--period", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--plain", action="store_true", help="Physics baseline without device reset machinery")
  parser.add_argument("--output", type=pathlib.Path)
  args = parser.parse_args()
  if not 0 <= args.switch_worlds <= args.nworld or args.period < 1 or args.repeats < 1:
    parser.error("Invalid switch count, period, or repeat count")
  wp.init()
  models, _ = make_models()
  assignments = np.arange(args.nworld, dtype=np.int32) % 3
  # Any cyclic rotation of this benchmark's worlds fits these capacities.
  capacity = (args.nworld + 2) // 3
  worlds = HeterogeneousWorlds(
    models,
    assignments,
    device="cuda:0",
    capacity_per_model={i: capacity for i in range(3)},
    data_kwargs={i: {"nconmax": 16, "njmax": 64} for i in range(3)},
  )
  tick = wp.zeros(1, dtype=int, device=worlds.device)
  reset_due = wp.zeros(1, dtype=int, device=worlds.device)
  reset_masks = [wp.ones(group.data.nworld, dtype=bool, device=worlds.device) for group in worlds.groups]
  requests = None if args.plain else worlds.enable_device_resets()

  def episode_logic(batch):
    if args.plain:
      wp.launch(_reset_due, dim=1, inputs=[tick], outputs=[reset_due], device=batch.device)
    else:
      wp.launch(
        _request_switches,
        dim=args.nworld,
        inputs=[tick, batch.world_model_ids, args.period, args.switch_worlds],
        outputs=[requests],
        device=batch.device,
      )
    wp.launch(_advance_tick, dim=1, outputs=[tick], device=batch.device)

  def controls(batch):
    if args.plain:

      def reset_standard():
        for group, mask in zip(batch.groups, reset_masks):
          io.reset_data(group.model, group.data, reset=mask)

      wp.capture_if(reset_due, reset_standard)
    for group in batch.groups:
      group.data.ctrl.fill_(0.05)

  worlds.capture(steps_per_launch=10, before_reset=episode_logic, before_step=controls)
  graph = worlds.graph
  pointers = [group.data.qpos.ptr for group in worlds.groups]
  durations = []
  # A continuous rollout: every reset request and model decision is generated on the GPU.
  # Host work is graph submission and timing events, with one synchronization at each sample's end.
  for sample in range(5 + args.repeats):
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(20):
      worlds.step()
    wp.record_event(end)
    duration = wp.get_event_elapsed_time(start, end) / 1000
    if sample >= 5:
      durations.append(duration)
  assert worlds.graph is graph
  assert [group.data.qpos.ptr for group in worlds.groups] == pointers
  finite = all(np.isfinite(group.data.qpos.numpy()).all() for group in worlds.groups)
  overflow = any(np.any(group.data.overflow.numpy()) for group in worlds.groups)
  status = [] if args.plain else worlds.device_reset_status.numpy().tolist()
  assert finite and not overflow and not any(status)
  report = {
    "nworld": args.nworld,
    "switch_worlds": 0 if args.plain else args.switch_worlds,
    "switch_period_steps": args.period,
    "episode_length": 200,
    "device_resets_enabled": not args.plain,
    "active_environment_steps_per_second": args.nworld * 200 / float(np.median(durations)),
    "batch_step_ms": float(np.median(durations)) * 1000 / 200,
    "sample_seconds": durations,
    "memory": worlds.memory_report(),
    "device_reset_status": status,
    "committed_world_resets": 0 if args.plain else int(worlds.device_reset_epoch.numpy().sum()),
    "finite_state": finite,
    "overflow": bool(overflow),
    "graph_reused": True,
    "notes": "Reset decisions, capacity checks, reassignment and initialization execute inside every captured step. "
    "No host reset calls or mapping readbacks occur during rollout. Readbacks above inspect final results only. "
    "Timings include device maintenance and reset work; physical capacities are fixed at startup. "
    "Every 200 steps all episodes reset to their current model on GPU, including the plain baseline.",
  }
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()

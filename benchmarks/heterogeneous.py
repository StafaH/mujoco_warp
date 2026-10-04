# Copyright 2026 The Newton Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Compare model buckets with original mjwarp and the sleep-policy PR 1680 union.

Run each mode in a fresh process. The sleep backend must contain PR 1680:
  uv run benchmarks/heterogeneous.py --mode grouped --nworld 8192 --output grouped.json
  uv run benchmarks/heterogeneous.py --mode sleep --mjwarp-path ../mjwarp-sleep-baseline \
    --nworld 8192 --output sleep.json
  uv run benchmarks/heterogeneous.py --mode original --model humanoid --output original.json
  uv run benchmarks/heterogeneous.py --mode grouped --model humanoid --virtual --output virtual.json

Models are primitive seven-joint arms (single/dual, with a free box and table), and the
repository's 27-DOF humanoid. All share the same static floor and table. No policy is loaded.
"""

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time

import mujoco
import numpy as np
import warp as wp


def _arm_xml(prefix, x):
  bodies = []
  for i in range(7):
    pos = f"{x} 0 0.72" if i == 0 else "0 0 .16"
    axis = "0 0 1" if i % 2 == 0 else "0 1 0"
    bodies.append(
      f'<body name="{prefix}link{i}" pos="{pos}">'
      f'<joint name="{prefix}joint{i}" axis="{axis}" damping="1" armature=".1" range="-120 120"/>'
      '<geom type="capsule" fromto="0 0 0 0 0 .14" size=".035" mass=".5"/>'
    )
  actuators = "".join(f'<position joint="{prefix}joint{i}" kp="20"/>' for i in range(7))
  return "".join(bodies) + "</body>" * 7, actuators


def _arm_spec(dual):
  body, actuators = _arm_xml("left_", 1.5)
  if dual:
    other_body, other_actuators = _arm_xml("right_", 2.5)
    body += other_body
    actuators += other_actuators
  body += '<body name="box" pos="2 0 .86"><freejoint/><geom type="box" size=".06 .06 .06" mass=".2"/></body>'
  return mujoco.MjSpec.from_string(
    '<mujoco><default><geom condim="3" friction=".7 .005 .0001"/></default>'
    f"<worldbody>{body}</worldbody><actuator>{actuators}</actuator></mujoco>"
  )


def make_models(sleep=False):
  """Return separate scenes and their identical union, with disjoint dynamic trees."""
  humanoid = mujoco.MjSpec.from_file(str(pathlib.Path(__file__).parent / "humanoid/humanoid.xml"))
  humanoid.delete(humanoid.geom("floor"))
  humanoid.option.timestep = 0.002
  humanoid.visual.map.zfar = 50
  children = [_arm_spec(True), humanoid, _arm_spec(False)]
  base_xml = """
  <mujoco>
    <option timestep=".002" solver="Newton" jacobian="dense" iterations="10" ls_iterations="50"/>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 .1" condim="3"/>
      <geom name="table" type="box" pos="2 0 .35" size=".9 .6 .35" condim="3"/>
    </worldbody>
  </mujoco>
  """
  models = []
  for child in children:
    spec = mujoco.MjSpec.from_string(base_xml)
    spec.attach(child.copy(), prefix="robot_", frame=spec.worldbody.add_frame())
    models.append(spec.compile())
  union = mujoco.MjSpec.from_string(base_xml)
  for i, child in enumerate(children):
    union.attach(child.copy(), prefix=f"model{i}_", frame=union.worldbody.add_frame())
  union_model = union.compile()
  for model in [*models, union_model]:
    if sleep or model is union_model:
      model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    # Ensure active worlds keep doing work throughout the timing window.
    model.tree_sleep_policy[:] = int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER)
  return models, union_model


def _array_bytes(obj):
  """Count unique array storage referenced by model/data dataclasses, excluding driver caches."""
  allocations = {}

  def visit(value):
    if isinstance(value, wp.array) and value.ptr:
      key = (str(value.device), value.ptr)
      allocations[key] = max(allocations.get(key, 0), value.capacity)
    elif dataclasses.is_dataclass(value):
      for field in dataclasses.fields(value):
        visit(getattr(value, field.name))
    elif isinstance(value, dict):
      for item in value.values():
        visit(item)
    elif isinstance(value, (list, tuple)):
      for item in value:
        visit(item)

  visit(obj)
  return sum(allocations.values())


def _revision(path):
  return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=["grouped", "sleep", "original"], required=True)
  parser.add_argument("--model", choices=["dual_arm", "humanoid", "single_arm"], help="Use a homogeneous batch of this model")
  parser.add_argument("--mjwarp-path", type=pathlib.Path)
  parser.add_argument("--nworld", type=int, default=8192)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--unroll", type=int, default=10)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--warmup-samples", type=int, default=0, help="Untimed reset-and-rollout samples before measurement")
  parser.add_argument("--nconmax", type=int, default=64)
  parser.add_argument("--njmax", type=int, default=192)
  parser.add_argument("--group-sleep", action="store_true")
  parser.add_argument("--virtual", action="store_true")
  parser.add_argument("--output", type=pathlib.Path, required=True)
  args = parser.parse_args()
  if min(args.nworld, args.steps, args.unroll, args.repeats) < 1 or args.steps % args.unroll:
    parser.error("positive counts and steps divisible by unroll are required")
  if args.warmup_samples < 0:
    parser.error("--warmup-samples must be nonnegative")
  if args.mode == "original" and args.model is None:
    parser.error("--mode original requires --model")
  if args.virtual and args.mode != "grouped":
    parser.error("--virtual requires --mode grouped")
  if args.mjwarp_path:
    sys.path.insert(0, str(args.mjwarp_path.resolve()))
  import mujoco_warp as mjw

  # Allow the new runner to use exactly the PR's physics kernels for controlled comparisons.
  mjw.__path__.append(str(pathlib.Path(__file__).resolve().parents[1] / "mujoco_warp"))
  wp.init()
  wp.set_device("cuda:0")
  models, union = make_models(sleep=args.group_sleep)
  model_id = ("dual_arm", "humanoid", "single_arm").index(args.model) if args.model else None
  ids = (
    np.full(args.nworld, model_id, dtype=np.int32)
    if model_id is not None
    else np.arange(args.nworld, dtype=np.int32) % len(models)
  )
  kwargs = dict(nconmax=args.nconmax, njmax=args.njmax)
  if args.mode == "grouped":
    from mujoco_warp.experimental import HeterogeneousWorlds

    extra = {"memory": "virtual"} if args.virtual else {}
    worlds = HeterogeneousWorlds(models, ids, data_kwargs={i: kwargs for i in range(3)}, **extra)
    pairs = [(g.model, g.data) for g in worlds.groups]
    step = worlds.step
  elif args.mode == "original":
    m = mjw.put_model(models[model_id])
    d = mjw.make_data(models[model_id], nworld=args.nworld, **kwargs)
    pairs = [(m, d)]

    def step():
      mjw.step(m, d)

  else:
    if not hasattr(mjw.SleepPolicy, "ALWAYS"):
      raise RuntimeError("The sleep baseline requires PR 1680; pass --mjwarp-path to its checkout")
    m = mjw.put_model(union, batch_sizes={"tree_sleep_policy": args.nworld})
    tree_owner = np.empty(union.ntree, dtype=np.int32)
    for tree in range(union.ntree):
      body = np.flatnonzero(union.body_treeid == tree)[0]
      tree_owner[tree] = int(mujoco.mj_id2name(union, mujoco.mjtObj.mjOBJ_BODY, int(body))[5])
    policies = np.where(ids[:, None] == tree_owner, int(mjw.SleepPolicy.AUTO_NEVER), int(mjw.SleepPolicy.ALWAYS))
    m.tree_sleep_policy = wp.array(policies, dtype=int)
    d = mjw.make_data(union, nworld=args.nworld, nvmax=max(x.nv for x in models), **kwargs)
    pairs = [(m, d)]

    def step():
      mjw.step(m, d)

  # Fixed nonzero commands, different per robot, exercise actuation as well as contacts.
  def reset():
    for m, d in pairs:
      mjw.reset_data(m, d)
      d.ctrl.fill_(0.05)

  reset()
  started = time.perf_counter()
  step()
  wp.synchronize()
  with wp.ScopedCapture() as capture:
    for _ in range(args.unroll):
      step()
  graph = capture.graph
  for _ in range(3):
    wp.capture_launch(graph)
  wp.synchronize()
  for _ in range(args.warmup_samples):
    reset()
    for _ in range(args.steps // args.unroll):
      wp.capture_launch(graph)
    wp.synchronize()
  compile_seconds = time.perf_counter() - started
  timings = []
  host_timings = []
  for _ in range(args.repeats):
    reset()
    wp.synchronize()
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    begin = time.perf_counter()
    wp.record_event(start)
    for _ in range(args.steps // args.unroll):
      wp.capture_launch(graph)
    wp.record_event(end)
    timings.append(wp.get_event_elapsed_time(start, end) / 1000)
    host_timings.append(time.perf_counter() - begin)

  snapshots = {}
  if args.mode == "grouped":
    for group in worlds.groups:
      snapshots[f"qpos_{group.model_id}"] = group.data.qpos.numpy()[0]
      snapshots[f"qvel_{group.model_id}"] = group.data.qvel.numpy()[0]
  elif args.mode == "original":
    snapshots[f"qpos_{model_id}"] = d.qpos.numpy()[0]
    snapshots[f"qvel_{model_id}"] = d.qvel.numpy()[0]
  else:
    qs, vs = d.qpos.numpy(), d.qvel.numpy()
    q, v = 0, 0
    for model_id, model in enumerate(models):
      row = np.flatnonzero(ids == model_id)
      if len(row):
        snapshots[f"qpos_{model_id}"] = qs[row[0], q : q + model.nq]
        snapshots[f"qvel_{model_id}"] = vs[row[0], v : v + model.nv]
      q, v = q + model.nq, v + model.nv
  overflow = [int(np.bitwise_or.reduce(d.overflow.numpy())) for _, d in pairs]
  finite = all(np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all() for _, d in pairs)
  if any(overflow) or not finite:
    raise RuntimeError(f"Invalid benchmark: non-finite state or capacity overflow: {overflow}")
  seconds = float(np.median(timings))
  package_root = pathlib.Path(mjw.__file__).parents[1]
  result = {
    "mode": args.mode,
    "model_selection": args.model,
    "virtual": args.virtual,
    "group_sleep": args.group_sleep,
    "device": wp.get_device().name,
    "warp_version": wp.__version__,
    "mujoco_version": mujoco.__version__,
    "mjwarp_revision": _revision(package_root),
    "mjwarp_path": str(package_root),
    "warp_path": str(pathlib.Path(wp.__file__).parents[1]),
    "nworld": args.nworld,
    "model_counts": np.bincount(ids, minlength=3).tolist(),
    "models": [{k: int(getattr(m, k)) for k in ("nq", "nv", "nu", "nbody", "ngeom")} for m in models],
    "union": {k: int(getattr(union, k)) for k in ("nq", "nv", "nu", "nbody", "ngeom")},
    "nconmax": args.nconmax,
    "njmax": args.njmax,
    "steps": args.steps,
    "unroll": args.unroll,
    "warmup_samples": args.warmup_samples,
    "seconds_samples": timings,
    "host_seconds_samples": host_timings,
    "env_steps_per_second": args.nworld * args.steps / seconds,
    "milliseconds_per_batch_step": 1000 * seconds / args.steps,
    "model_bytes": sum(_array_bytes(m) for m, _ in pairs),
    "data_bytes": sum(_array_bytes(d) for _, d in pairs),
    "compile_seconds": compile_seconds,
    "max_constraints": [int(d.nefc.numpy().max()) for _, d in pairs],
    "contacts": [int(d.nacon.numpy()[0]) for _, d in pairs],
    "overflow": overflow,
    "all_states_finite": finite,
  }
  if args.mode == "grouped":
    result["memory_report"] = worlds.memory_report()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  np.savez(args.output.with_suffix(".npz"), **snapshots)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

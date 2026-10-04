"""Measure captured rollout times for DynamicTable examples at 8,192 worlds.

Run: uv run --no-sync python benchmarks/dynamic_table_performance.py
"""

import gc
import json
from pathlib import Path

import numpy as np
import warp as wp
from warp.examples.core.example_dynamic_collision_worlds import simulate

from mujoco_warp.experimental.dynamic_contacts_example import OBJECTS
from mujoco_warp.experimental.dynamic_contacts_example import _model
from mujoco_warp.experimental.dynamic_contacts_example import _run


def main():
  wp.init()
  worlds = 8192
  visible = [0, 1, 2, 64, 65, 128, 129, 192]
  repeats = 7
  measurements = {
    "pure_warp": {"dense": [], "table": []},
    "mjwarp": {"generous_native": [], "tight_native": [], "tight_table_only": [], "tight_table_recording": []},
  }
  # Alternate execution order to reduce consistent warm-device ordering bias.
  for trial in range(repeats):
    runs = {}
    for packed in [False, True] if trial % 2 == 0 else [True, False]:
      value = simulate(worlds, visible, packed, wp.get_device("cuda:0"))
      measurements["pure_warp"]["table" if packed else "dense"].append(value["memory"]["rollout_ms"])
      runs[packed] = value
    np.testing.assert_array_equal(runs[False]["counts"], runs[True]["counts"])
    for frame in range(len(runs[False]["positions"])):
      for view in range(len(visible)):
        n = runs[False]["counts"][frame, view]
        np.testing.assert_allclose(
          runs[False]["positions"][frame, view, :n], runs[True]["positions"][frame, view, :n], atol=0.003, rtol=0.003
        )
    del runs, value
    gc.collect()
    print(f"Pure Warp trial {trial + 1}/{repeats} complete", flush=True)
  models = [_model(n) for n in OBJECTS]
  ids = np.arange(worlds, dtype=np.int32) % len(OBJECTS)
  configs = [
    ("generous_native", False, False),
    ("tight_native", True, False),
    ("tight_table_only", True, False),
    ("tight_table_recording", True, True),
  ]
  for trial in range(repeats):
    reference = None
    for name, tight, record in configs if trial % 2 == 0 else configs[::-1]:
      metrics, state, *_ = _run(models, ids, tight=tight, record=record, snapshot=name == "tight_table_only")
      measurements["mjwarp"][name].append(metrics["rollout_ms"])
      if reference is None:
        reference = state
      else:
        for a, b in zip(reference, state, strict=True):
          np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
      gc.collect()
    print(f"Mjwarp trial {trial + 1}/{repeats} complete", flush=True)
  summary = {
    scene: {
      name: {"median_ms": float(np.median(values)), "min_ms": min(values), "max_ms": max(values), "samples_ms": values}
      for name, values in variants.items()
    }
    for scene, variants in measurements.items()
  }
  result = {
    "worlds": worlds,
    "device": wp.get_device("cuda:0").name,
    "repeats": repeats,
    "pure_warp_steps": 256,
    "mjwarp_steps": 384,
    "summary": summary,
    "method": "CUDA event timings for complete captured rollouts; compilation/allocation/readback excluded; alternating variant order; both pure Warp variants record GPU frames; mjwarp isolates table-only overhead and separately adds recording; state parity checked each trial.",
  }
  path = Path(__file__).with_name("heterogeneous_results") / "dynamic_table_performance.json"
  path.write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

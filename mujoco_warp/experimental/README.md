# Heterogeneous worlds and dynamic memory prototype

For an illustrated introduction with interactive reset and benchmark examples, open the [standalone HTML report](heterogeneous_report.html).
For API contracts, native implementation details, and integration examples, open the [HTML engineering report](heterogeneous_engineering_report.html), which covers Warp first and mjwarp second.
For the table itself, open the [DynamicTable design and implementation report](dynamic_table_report.html). The [real mjwarp collision recording](dynamic_contacts_demo.html) shows changing contacts across 8,192 worlds, with measured memory and rollout results.

This prototype runs different MuJoCo models in one CUDA graph and lets GPU episode logic choose a different model for each logical world at reset. The GPU path performs request validation, slot assignment, state reset and action routing inside the captured graph, with no reset readback, host synchronization or recapture. The accompanying Warp fork provides dynamic component tables and CUDA memory primitives.

Two GPU backends are available: fixed model batches for a known capacity envelope, and a pooled workspace for unrestricted population changes within the startup model catalog. Both keep reset work on device. The host-reset prototype has been removed. Inserting bodies into an existing model during a running graph is not implemented.

## Saved checkpoint

The mjwarp work is saved on [`stafah/heterogeneous-worlds-prototype`](https://github.com/StafaH/mujoco_warp/tree/stafah/heterogeneous-worlds-prototype).
Its companion Warp branch is [`stafah/dynamic-memory-prototype`](https://github.com/StafaH/warp/tree/stafah/dynamic-memory-prototype), with initial checkpoint revision `0b117d81eaab9eadf66dfa1d4ed467c3fe483441`.
Both checkouts are needed; the mjwarp editable dependency expects the Warp checkout at `../mustafa_warp`.
Native libraries, Python environments, and kernel caches are not committed, so rebuild Warp and synchronize mjwarp after cloning.

```bash
git clone --branch stafah/dynamic-memory-prototype git@github.com:StafaH/warp.git mustafa_warp
git clone --branch stafah/heterogeneous-worlds-prototype git@github.com:StafaH/mujoco_warp.git mustafa_mjwarp
cd mustafa_warp
uv run build_lib.py --quick
cd ../mustafa_mjwarp
uv sync --extra dev --inexact
uv run --no-sync python -m mujoco_warp.experimental.native_contacts_example --worlds 8192 --repeats 7
```

The quick build requires a compatible driver/toolkit; use the standard `build_lib.py` build otherwise.
The checkpoint includes prototype source, tests, benchmark samples, visual recordings, HTML reports, and the sharing ZIP.
The latest direct contact path is `NativeContactTable`, consumed by native collision detection and the native solver without a snapshot copy.
The generic Warp `DynamicTable` remains a separate ECS storage primitive with handles and compaction.
Physical capacity is committed before capture, shared native contacts are transient between complete model batches, and the tested direct path reduces memory while running at approximately native speed.
Validation at this checkpoint passed 16 Warp feature tests and 52 mjwarp experimental tests, plus the repository commit hooks.
Further work should measure consumer and solver changes before claiming a native physics speedup, and preserve the GPU-only reset path and explicit overflow handling.

## Run the prototype

The repository's `pyproject.toml` links `warp-lang` to `../mustafa_warp` as an editable dependency. Build that fork before synchronizing this environment:

```bash
cd ../mustafa_warp
uv run build_lib.py --quick
cd ../mustafa_mjwarp
uv sync --extra dev --inexact
uv run --no-sync python -m mujoco_warp.experimental.pooled_example
```

The quick build requires a driver at least as new as the installed CUDA toolkit. The tested machine has CUDA toolkit 13.2 and driver 595.91.07. The pooled example changes all worlds from a mixed robot population to one robot type, then another, then back to a mixed population. Episode decisions and action routing execute on GPU under the same graph. `heterogeneous_device_example` separately demonstrates fixed-capacity GPU slots with an uninterrupted third world.

For actual robots, supply any models supported by MuJoCo Warp:

```python
import mujoco
import numpy as np
from mujoco_warp.experimental import HeterogeneousWorlds

models = [mujoco.MjModel.from_xml_path(path) for path in
          ["dual_robot_table.xml", "humanoid.xml", "single_robot.xml"]]
worlds = HeterogeneousWorlds(
    models,
    np.arange(8192, dtype=np.int32) % 3,
    device="cuda:0",
    memory="default",  # VMM is optional; grouping and reset reassignment do not require it.
    data_kwargs={i: {"nconmax": 64, "njmax": 192} for i in range(3)},
)
worlds.capture()
worlds.world_array(2, "ctrl").fill_(0.1)
worlds.step()
print(worlds.memory_report())
```

Choose contact and constraint capacities for the actual task; the example values are not universal guarantees. `world_array()` retains a leading world dimension. `arrays("qpos")` returns `(logical_world_ids, values)` for each model group, and both are device arrays. These are zero-copy views. Different robot widths are represented as separate tensors; a single rectangular tensor would require padding or packing. Standard Warp DLPack/PyTorch interop applies, with the usual ownership and stream synchronization requirements. DLPack ownership was tested; PyTorch itself was not installed in the test environment.

## Direct shared native contacts

```bash
uv run --no-sync python -m mujoco_warp.experimental.native_contacts_example \
  --worlds 8192 --repeats 7 --output mujoco_warp/experimental/native_contacts_demo.html
```

The [direct native collision demo](native_contacts_demo.html) replaces each model group's independent contact allocations with prefix views of one `NativeContactTable`. Collision detection writes directly into those columns, and the native constraint builder and solver read the same values. There is no downstream contact snapshot, payload copy, or ECS compaction. `io.make_data(contact_storage=owner)` skips the original contact allocations at construction rather than rebinding already allocated arrays. `HeterogeneousWorlds(contact_storage=owner)` steps complete model batches sequentially and offers `capture(after_group=callback)` for consumers before the next batch overwrites contacts.

This is transient contact storage using Warp's stable-address memory primitive. It deliberately omits the generic DynamicTable's entity handles and compaction scratch: contacts already use bounded native append counters and are rebuilt each step, and native solver indices depend on the existing row order. Capacity is committed at startup. No host reset, readback, allocation, or recapture occurs in the loop. `memory="default"` provides the same pooling with ordinary allocation; virtual memory is optional. The native table packs all columns into one arena to avoid a page minimum for every small column.

Contacts are valid only until another batch reuses the table. Consume them in `after_group`; the wrapper rejects persistent `arrays("contact.*")` inspection in this mode. Do not interleave different groups' `step1`/`step2`, call `io.reset_data` against stale shared contacts, or disable collision regeneration. The wrapper rejects `run_collision_detection=False`. Sleeping's incremental collision passes, sensors, and integration finish before reuse. Native contact and broadphase overflow are observed on device after each batch and retained as sticky errors for inspection after rollout.

At 8,192 worlds, the independent native contact arrays occupy 28,962,960 bytes. The shared table occupies 20,443,524 bytes with ordinary allocation or 20,971,524 bytes with virtual memory, including its error flag. The unchanged per-group capacities are 10,924, 43,696, and 131,040 rows: sharing needs the maximum (131,040), rather than their sum (185,660), because groups run sequentially. This preserves each group's capacity but no longer preserves all groups' contacts simultaneously. Total native physics arrays are 220,103,173 bytes independently, 211,583,737 bytes with ordinary pooling, and 212,111,737 bytes with virtual pooling. Recording is separate.

Seven alternating-order trials compare equal physics work with recording disabled. The demo JSON and table report contain every timing sample. The direct path runs at approximately native speed in this scene and removes the earlier snapshot's measured slowdown; it does not demonstrate faster native physics. Final generalized positions and velocities, all recorded object positions, and per-world contact counts match the independent baseline. New direct-storage coverage plus existing heterogeneous regressions passed 27 tests; 20 selected core allocation/reset tests also passed.

## Visual dynamic contacts: historical snapshot baseline

```bash
uv run --no-sync python -m mujoco_warp.experimental.dynamic_contacts_example \
  --worlds 8192 --output mujoco_warp/experimental/dynamic_contacts_demo.html
```

Open the generated HTML directly in a browser. The recording uses native mjwarp sphere/plane collision detection and constraint solving across worlds with 1, 4, or 12 spheres. GPU kernels reset the drops and rebuild a contact observation table inside one unchanged graph. Only after the 384-step rollout are recorded positions and contact points copied to the host. Objects and model assignments stay fixed in this example; contact populations change. The HTML provides playback, seeking, contact normals, memory comparisons, and a contact-count timeline. The sibling JSON contains the measured results and recording.

`ContactTable.rebuild(groups)` clears its previous rows, counts native contacts by logical world, reserves all rows with `DynamicTable.append_worlds`, scatters contact components using per-world cursors, and compacts to publish world ranges. Components are position, normal, signed penetration depth, and model-local geometry IDs. The adapter checks native broadphase/contact overflow and table exhaustion; partial snapshots must be rejected. Its `check_errors()` method is an inspection boundary outside capture. Table physical capacity is committed before capture.

Native mjwarp already packs contacts across a model batch with a global `naconmax` capacity. The demo therefore compares generously provisioned native contacts, tighter native contacts, and the tighter version plus the table. The table is a downstream snapshot for GPU consumers and export: it adds storage and copying and does not replace the solver's contact arrays. Savings shown for native physics come from capacity tuning. Tight capacity is specific to this tested scene, not a guarantee for arbitrary robots or collision bursts. Per-world constraint capacity remains separate.

At 8,192 worlds, measured native array storage was 354,718,069 bytes with generous contacts and 220,103,173 bytes with the tighter budget. The contact table plus its adapter workspace added 48,562,240 bytes; GPU recording added 2,450,688 bytes separately. Contact counts ranged from 0 to 46,415. Final generalized positions matched both native baselines exactly. These byte totals exclude CUDA context, graphs, compiled kernels, allocator caches, and small reset/validation arrays. The report includes the individual rollout timings; they are illustrative single measurements rather than a controlled throughput benchmark.

Repeated performance measurements are available with `uv run --no-sync python benchmarks/dynamic_table_performance.py`. Seven alternating-order rollouts at 8,192 worlds measured median pure-Warp times of 62.35 ms for fixed per-world buffers and 40.10 ms for the packed-table version (256 steps). This 1.55× speedup comes from processing and clearing a smaller capacity under a workload-specific global budget; it does not isolate virtual-memory allocation from packing. For mjwarp, tight native physics took 344.73 ms versus 395.84 ms with the contact snapshot alone (384 steps, no recording in either case), a 14.8% slowdown. The current snapshot adds work to already packed native contacts. All samples and methodology are in `benchmarks/heterogeneous_results/dynamic_table_performance.json`, and the table report explains capacity differences and interpretation.

`world_group` and `world_row` map stable logical IDs to compact physical rows. Groups are identified by the model ID supplied by the caller: two models with equal dimensions can still have different joints, collision geometry, or actuator semantics, so the implementation does not infer equivalence from array sizes. `capture(before_step=callback)` can include device policy/action routing in the graph. Updating existing array contents is supported; replacing graph-referenced arrays is not.

## GPU reset loop

Call `requests = worlds.enable_device_resets()` once before capture. A device episode kernel writes a destination model ID into `requests[world]`; `-1` means no reset. Then capture with `before_reset=episode_logic` and `before_step=route_actions`. Each captured timestep executes:

1. GPU episode logic reads the preceding state and writes reset requests.
2. GPU kernels validate the whole request batch and check destination capacities after selected source slots are released.
3. On success, slots and logical mappings change, selected state is initialized from startup defaults, and action routing sees the new model assignment.
4. Physics recomputes contacts, constraints and observations for the new state.

Every successful reset increments `device_reset_epoch[world]` and consumes its request. Invalid model IDs or insufficient capacity reject the entire transaction without changing mappings, epochs or simulation state. Requests remain pending so device logic can replace them or retry after another world releases capacity. `device_reset_status` contains sticky error bits: 1 for invalid/unprovisioned models and 2 for capacity exhaustion. `device_reset.transaction_status` reports the current transaction's error bits and committed request count. Reading these arrays on the host is optional diagnostic work outside the execution loop.

The model catalog, physical capacities and reset defaults are fixed during setup. Provision initially unused models with `capacity_per_model`. A same-model request also resets its episode. Unselected worlds keep their slots and state. There is no host fallback when capacity runs out, and no physical page commitment during replay.

`world_group`, `world_row`, `world_model_ids` and each group's `world_ids` are device mappings. `arrays()` exports physical group tensors; `-1` marks a vacant row. Host `locate()` and `world_array()` are disabled after enabling GPU resets, since cached host mappings would be stale. `inspect_layout()` and `memory_report()` explicitly read back diagnostic information and belong outside the loop.

Reset maintenance runs immediately before action routing and physics. Read previous contact observations in `before_reset`, or new observations after physics; `before_step` must not consume contacts left from the previous layout. Model defaults and workspace arrays are allocated only at setup. Captured conditional nodes skip per-model reset kernels when that model has no changed slots.

The focused GPU tests replace host readback and synchronization functions with failures during capture and replay. They verify GPU-generated swaps, unchanged graph/data pointers, ordinary/VMM allocation, activation of initially unused models, atomic capacity failure, retry, and invalid-request handling.

## GPU population changes with shared workspace

`PooledHeterogeneousWorlds` removes per-model capacity limits. Every logical world can choose any model in the startup catalog, including changing all 8192 worlds to one model. A GPU worklist groups worlds by their current model and schedules fixed-size chunks through one shared physics workspace. Empty model populations skip their physics branches. The CUDA graph, persistent state pointers and physical allocations remain unchanged.

Persistent integration state is stored by logical world ID. The larger collision and solver arrays occupy one working batch, whose different model layouts alias the same storage. Each GPU chunk gathers its worlds' state, runs the existing specialized mjwarp physics, and scatters updated state and requested observations. Device conditional loops process however many chunks each model currently needs. This avoids allocating 8192 complete Data rows for every possible robot.

```python
from mujoco_warp.experimental import PooledHeterogeneousWorlds

worlds = PooledHeterogeneousWorlds(
    models,                          # Immutable catalog: dual arm, humanoid, single arm.
    np.arange(8192, dtype=np.int32) % 3,
    batch_size=1024,                  # Working rows reused across all models.
    scratch_bytes=64 << 20,           # Capture scratch; choose for model complexity.
    device="cuda:0",
    data_kwargs={i: {"nconmax": 16, "njmax": 64} for i in range(3)},
)
# Both callbacks enqueue GPU kernels. Episode logic writes reset_model_ids[world].
worlds.capture(before_reset=episode_logic, before_step=policy, steps_per_launch=10)
for _ in range(1000):
    worlds.step()                    # Replay only: no host reset, readback, or recapture.
```

The request array uses model IDs with `-1` for no reset. Successful requests are consumed and increment per-world epochs. Invalid model requests reject the whole transaction; there is no host fallback. Physics continues advancing the existing episodes after a rejected request. The logical world count and model catalog are fixed at startup, but their population mix is unrestricted.

`worlds.state` exposes stable, zero-copy persistent arrays such as `qpos`, `qvel`, `ctrl`, `time`, `history` and `sensordata`. A field uses the largest catalog model width; only the current model's prefix is meaningful. This introduces padding in the small persistent state, while sharing the much larger physics workspace. Add world-indexed outputs through `observation_fields`; transient contact buffers are not exported as persistent tensors. Policy code must select the model-appropriate observation and action widths.

`wp.ScopedCaptureScratch` is the additional Warp mechanism this backend needs. It provides preallocated storage to both Python array creation and native scratch users, such as CUB scans, while the graph is constructed. Conditional physics branches can then execute without allocation nodes. Serial model branches reuse the same scratch addresses. Callback-created scratch temporaries must finish their work before physics reuses that storage; callbacks cannot install those temporaries as persistent state.

All physical pages are committed at startup. CUDA VMM provides the shared arena's stable allocation and overlapping model views; the algorithm could also use a sufficiently sized ordinary allocation. It does not perform GPU page-fault allocation or call a host mapping service during replay. Scratch exhaustion is a setup error: increase `scratch_bytes` or reduce `batch_size` before running the graph.

The pooled prototype rejects sleeping and model callbacks. Sleeping kernels retain derived body/solver caches that would need additional persistent storage; global policy callbacks operate through the persistent state instead. Shared model parameters and reset defaults remain fixed after setup. The final partial chunk of each model still simulates unused rows, so batch size affects both memory and wasted work. Sticky per-world `physics_overflow` and `physics_nonfinite` diagnostics accumulate across workspace reuse.

Correctness tests cover all-world population changes, invalid-request transactions, stable graph/state pointers, multiple chunks, all four integrators, contacts on translated/rotated static planes, activation and sensor delays, mocap/user data, and applied forces. Dense constraint padding is cleared before each chunk because aliased layouts can overwrite padding that ordinary fresh Data keeps zero.

## GPU-only measurements at 8192 worlds

These measurements use the same dual-arm, humanoid and single-arm models and RTX PRO 6000 described below, with 16 contacts and 64 constraint rows per working world. GPU episode logic resets episodes every 200 steps. The graph contains ten timesteps per launch. Each population stage has five untimed warmup rollouts and five timed 200-step rollouts; results are medians measured with CUDA events. Device planning, gathering/scattering, state initialization and physics are all inside the timed graph.

The pooled benchmark executes one continuous graph schedule: mixed robots → all humanoids → all single arms → mixed robots again. No host writes model requests, changes mappings, allocates pages, or recaptures between stages. GPU-recorded model histograms verify each population. All runs retain graph and state-array pointers; sticky diagnostics report zero overflow and zero nonfinite worlds across every chunk, including earlier stages.

| Working batch size | Data, state, models and mappings, MB | Preallocated scratch, MB | Total owned memory, MB | Mixed, million env steps/s | All humanoids | All single arms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 41.6 | 67.1 | 108.7 | 2.555 | 2.953 | 2.886 |
| 2731 | 98.2 | 151.0 | 249.2 | 6.416 | 6.208 | 7.206 |
| 4096 | 142.3 | 218.1 | 360.4 | 5.490 | 7.346 | 9.981 |

The 2731-row workspace fits this balanced three-model distribution almost exactly and provides the best mixed throughput among these tested sizes. The 4096-row workspace processes a homogeneous 8192-world population in two chunks, improving homogeneous throughput at higher memory cost. The 1024-row workspace minimizes memory but pays for many more physics-pipeline iterations. These are measured tradeoffs, not an automatic tuning policy.

**Memory accounting differs from the older tables below:** pooled totals include the entire preallocated capture-scratch budget, while the older fixed-batch and sleep-union numbers exclude graph scratch. CUDA context/JIT/graph metadata and allocator caches remain excluded from both. The persistent logical-world state itself is only 7.63 MB here. The largest actual scratch use was 53.42, 142.48 and 213.69 MB for the three workspace sizes respectively. Scratch is allocated at startup; unused budget remains resident. A 4 GiB virtual reservation does not mean 4 GiB is physically committed.

The pooled path has a throughput cost relative to an ordinary homogeneous batch: it gathers/scatters state, initializes reused workspace and runs chunk loops. For comparison, the original API below reached 11.872 million steps/s for all humanoids and 17.984 million for all single arms, with fixed populations. Those earlier runs excluded reset maintenance and used their own warmup protocol. The pooled results establish fully GPU-driven flexibility; they do not establish a homogeneous speedup.

For a known population-capacity envelope, fixed model batches remain faster. Their GPU-reset benchmark measured 7.108 million steps/s for the ordinary physics baseline with periodic GPU episode resets, 6.940 with the GPU reassignment machinery enabled but no type changes, 6.662 while changing 96 worlds' robot types every ten steps, and 4.829 while changing those 96 worlds every step. Total array memory was 202.67 MB with reassignment enabled, excluding graph scratch. The idle machinery adds about 2.4% to step time. These fixed batches reject per-model capacity overflow on device; they cannot replace the pooled path for unrestricted population changes.

Reproduce the pooled tradeoff with:

```bash
uv run --no-sync benchmarks/heterogeneous_pooled.py --batch-size 2731 --scratch-mib 144 \
  --output benchmarks/heterogeneous_results/pooled_2731_8192.json
uv run --no-sync benchmarks/heterogeneous_device_reset.py --switch-worlds 96 --period 10 \
  --output benchmarks/heterogeneous_results/device_reset_96_period10_8192.json
```

Raw results are [`pooled_1024_8192.json`](../../benchmarks/heterogeneous_results/pooled_1024_8192.json), [`pooled_2731_8192.json`](../../benchmarks/heterogeneous_results/pooled_2731_8192.json), [`pooled_4096_8192.json`](../../benchmarks/heterogeneous_results/pooled_4096_8192.json) and `device_reset_*.json` in the same directory.

## What was implemented

| Layer | Mechanism | Execution boundary |
| --- | --- | --- |
| Warp `VirtualMemory` | CUDA address reservation, page commitment, zero initialization, stable pointers, owned array views | Physical growth on the host between completed replays |
| Warp `DynamicTable` | Column storage, device counts, stable entity handles and generations, bounded append, delete, compaction, world ranges | Logical changes and grouping inside a captured graph |
| mjwarp `HeterogeneousWorlds` | Fixed model batches, GPU reset requests, stable logical mappings | All reset work inside the existing graph; fixed per-model capacities |
| mjwarp `PooledHeterogeneousWorlds` | Persistent state, GPU model worklists, shared physics workspace and conditional chunk loops | Arbitrary population mix inside one unchanged graph |
| Warp `ScopedCaptureScratch` | Shared preallocated Python/native temporary storage for conditional graph bodies | Setup-time suballocation; no allocation during replay |
| mjwarp virtual arena | All `Data` arrays allocated from one VMM arena per model group | Commits pages during initialization; retains storage until arena release |

The arena uses Warp's allocator protocol. It reserves 4 GiB per group by default, commits only the needed prefix, and aligns arrays to 256 bytes. Model arrays retain the ordinary allocator. Freed individual allocations are not reused; `arena_highwater_bytes` includes temporary allocations made during initialization. `memory_report()` includes this cost and page rounding rather than reporting only live array sizes. `memory="default"` is also supported and is slightly more memory efficient for a fixed workload.

`DynamicTable` is in `warp.experimental.table`. `append_worlds(counts)` scans per-world counts with 64-bit accumulation and reserves the whole batch once. It either succeeds completely or returns invalid offsets and a sticky error. Small device-side operations can use `append_row`/`append_rows`. Deletion invalidates a handle by incrementing its generation; compaction updates surviving handle locations and groups rows by world. Component columns are gathered through scratch and copied back so their exported addresses remain unchanged. Ordering within a world is unspecified.

Overflow never grants access to uncommitted rows. Applications must check errors, grow capacity, and explicitly retry the failed work. Do not replay a partially completed simulation step as an implicit retry. Table mutations are ordered phases on one stream. Full reserved-shape views must only be used by kernels that guard accesses with accessible counts; use a committed-length slice for ordinary tensor operations. GPU page faults are not handled automatically.

Madrona also stores component columns across worlds, sorts entities for locality, and maintains entity-to-row mappings. Its GPU growth path requests mapping work from a dedicated host thread; the host performs CUDA VMM operations and publishes completion. The paper's contiguous virtual storage enables tensor export without relocating component data. This prototype uses a simpler host boundary between graph replays instead of reproducing that synchronous GPU-to-host service. See the [Madrona paper, section 5](https://madrona-engine.github.io/shacklett_siggraph23.pdf) and the local source at `../madrona/src/mw/device/state.cpp`, `memory.cpp`, and `../madrona/src/mw/cuda_exec.cpp`.

The underlying reservation/mapping operations are described in NVIDIA's [CUDA VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html). The prototype adds explicit context synchronization and rejects mapping changes during capture. CUDA graph replay and CUDA graph capture are different phases: keeping and replaying the same graph permits host commitment between launches.

## Measurements at 8192 worlds

The workload has 2,731 dual-arm worlds, 2,731 humanoid worlds, and 2,730 single-arm worlds. Arms are seven-joint primitive capsule models, with a table and a free box; the humanoid is the repository's 27-DOF model. All scenes share the same static floor and table. The three scenes have 20, 27, and 13 velocity DOFs respectively; their union has 60. This measures physics under constant nonzero actions, not learned manipulation or locomotion policies, rendering, or observation networks.

Measurements use the NVIDIA RTX PRO 6000 Blackwell Workstation Edition, MuJoCo 3.12.1, and the locally built Warp 1.19.0.dev0. Each sample runs 200 steps of 0.002 seconds, using a graph with ten unrolled steps. Results are medians of five samples, measured with CUDA events after compilation and warmup. World state is reset before each sample. Each configuration runs in a separate process, without concurrent compute tests.

Both the union baseline and grouped configurations use the **same physics checkout**: base `eb912f1` with PR #1680 commit `c7f0625` cherry-picked. The experimental runner is loaded from this working tree. Consequently the comparison does not conflate the PR's physics changes with the new storage layout. The sleep baseline permanently disables unused trees and uses `nvmax=27`, following [PR #1680](https://github.com/google-deepmind/mujoco_warp/pull/1680). The grouped path normally disables sleeping because there are no unused robots to suppress.

| Configuration | Contacts per world | Constraint rows per world | Array or committed memory, MB | Million environment steps/s | ms per batch step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sleep union | 16 | 64 | 627.1 | 5.103 | 1.605 |
| Model groups, default allocator | 16 | 64 | 202.4 | 6.746 | 1.214 |
| Model groups, VMM arenas | 16 | 64 | 207.9 | 6.799 | 1.205 |
| Sleep union | 64 | 192 | 1162.4 | 4.729 | 1.732 |
| Model groups, default allocator | 64 | 192 | 424.5 | 6.577 | 1.246 |
| Model groups, VMM arenas | 64 | 192 | 436.4 | 6.587 | 1.244 |
| Model groups with sleeping enabled | 64 | 192 | 649.6 | 4.004 | 2.046 |

MB means 1,000,000 bytes. Memory includes model arrays, data arrays and world mapping; for virtual arenas it counts committed physical pages, including unused arena space. It excludes CUDA context/JIT/graph allocations and caching allocator pools. Twelve GiB of reserved virtual address space in the three arenas is not twelve GiB of physical memory.

For the tighter tested limits, the VMM grouped configuration uses **66.9% less reported memory** and delivers **1.33 times the throughput** of the union. Most savings come from eliminating inactive model state and union-sized workspaces. VMM preserves addresses and controls physical backing; it does not itself make physics faster. Keeping the sleep/compact solver path inside every group saves memory compared with the union but is slower here. These results do not establish a universal speedup for every topology distribution or capacity choice.

The workload reached at most 36 constraints in a world. All measured final states were finite and overflow flags were clear. Captured grouped dynamics also pass dedicated tests against independent mjwarp batches and MuJoCo. Across the saved benchmark snapshots, union/grouped final positions differ by up to approximately 0.00067, and velocities by approximately 0.0076, predominantly in the contacting box; these runs are not bitwise trajectory-equivalent. Snapshot files are retained for inspection, rather than claiming equivalence from throughput alone.

## Homogeneous comparison with original mjwarp

For a fixed homogeneous workload, the prototype has almost the same throughput as the original mjwarp API and uses slightly more memory. With one model, the runner creates one group and captures the same physics kernels as ordinary `put_model`, `make_data`, and `step`. There is no union-model padding to remove. The current VMM adapter commits all requested Data storage during construction; it does not yet grow contact or solver capacities from per-step demand.

The following comparison uses **8192 identical worlds per row**, on the same RTX PRO 6000. Both the original API and new runner use this fork's unchanged physics implementation and the same local Warp 1.19.0.dev0 build, isolating the method from compiler/runtime version changes. Sleeping is disabled in every configuration. Each run uses 16 contacts and 64 constraint rows per world, 200 simulated steps, ten steps per graph launch, five untimed full-rollout warmup samples, and the median of ten timed samples. Graph replay is timed identically in every configuration; Python construction/capture is excluded.

| Homogeneous scene | Original API, million steps/s | Grouped ordinary allocation, million steps/s | Grouped VMM, million steps/s | Original memory, MB | VMM memory, MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Humanoid | 11.872 | 11.871 | 11.838 | 265.7 | 268.6 |
| Dual primitive arm with box | 14.468 | 14.508 | 14.357 | 194.5 | 197.3 |
| Single primitive arm with box | 17.984 | 17.932 | 17.534 | 146.3 | 149.0 |

The ordinary allocator wrapper is within 0.3% of the original path in these runs. VMM throughput is 0.28%, 0.77%, and 2.51% lower respectively. These are observed end-to-end differences, not a measurement isolating the cost of memory translation. No dynamic table creation, deletion or sorting is performed during these homogeneous physics steps; that maintenance has its own cost when used.

The wrapper adds exactly 128 KiB of device world mappings at 8192 worlds. The VMM path adds approximately 2.8–2.9 MB in total, or 1.1–1.9%, including those mappings, page rounding, alignment and retained arena allocations. Memory includes nested model array lists as well as Data, but excludes CUDA runtime/JIT/graph overhead. Reserving a 4 GiB virtual arena does not commit 4 GiB physically.

As a separate check, the original API with the previous **Warp 1.15.0** package measured 11.777, 14.201, and 17.860 million steps/s for the humanoid, dual-arm, and single-arm scenes respectively. That package ran in an isolated environment using the same installed MuJoCo 3.12.1, NumPy version, physics sources and settings. These results support the same conclusion without attributing a runtime-version difference to the new allocator.

All twelve final configurations completed with finite state and no overflow flags. Representative saved position and velocity snapshots agree within approximately 1.7e-6 and 3.1e-5 respectively across the methods. The results and snapshots are named `homogeneous_{original,grouped,virtual,stock}_{model}_8192.{json,npz}` in the results directory. `original` uses the local Warp build; `stock` uses Warp 1.15.0. An initial run showed a clock ramp across samples, so these final results include the five full warmup rollouts.

For a fixed homogeneous training job, ordinary mjwarp allocation is the best default based on this prototype. The VMM machinery is useful when stable-address storage needs to grow; the heterogeneous grouping benefit appears when different worlds would otherwise reserve unused model state. The measured heterogeneous speedup should not be extrapolated to homogeneous workloads.

```bash
uv run --no-sync benchmarks/heterogeneous.py --mode original --model humanoid \
  --nworld 8192 --nconmax 16 --njmax 64 --repeats 10 --warmup-samples 5 \
  --output /tmp/homogeneous_original.json
uv run --no-sync benchmarks/heterogeneous.py --mode grouped --model humanoid --virtual \
  --nworld 8192 --nconmax 16 --njmax 64 --repeats 10 --warmup-samples 5 \
  --output /tmp/homogeneous_virtual.json
```

## Scaling and table maintenance

With 64 contacts and 192 constraint rows allocated per world:

| Worlds | Sleep union, million steps/s | Model groups, million steps/s | Sleep union memory, MB | Model group memory, MB |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.230 | 0.161 | 18.3 | 6.7 |
| 1024 | 1.537 | 1.057 | 145.4 | 53.1 |
| 8192 | 4.729 | 6.577 | 1162.4 | 424.5 |

Grouping runs a separate specialized physics pipeline for each model. A single captured graph removes repeated Python submission, but it does not fuse those pipelines. This explains the small-batch tradeoff and limits the current design when nearly every world has a different model. Bucketing a few robot families is the intended fast path.

The separate Warp ECS example creates, initializes, clears and groups rows inside one graph at 8192 worlds. It measured **0.079 ms for 57,344 rows** and **0.129 ms for 114,695 rows**, with **22.3 MiB** of table-owned storage including scratch. These are synthetic component rows, not a physics solver benchmark. The same graph and component address serve both row distributions. The initial per-row contended atomic approach took tens of milliseconds; batched scan/reservation removed that bottleneck.

This GPU's VMM granule is 2 MiB. Independent narrow component columns and their scratch arrays therefore have substantial minimum costs, even for a few rows. The mjwarp arena amortizes page rounding across hundreds of Data arrays. The table prototype still reserves each column independently to permit contiguous independent growth; pooling small metadata is a remaining optimization. Physical commitment latency is outside the steady-state timing. There is no physical shrinking or page reclamation before reservation destruction.

## Remaining physics work

The implemented pooled runner solves reset-time model reassignment within a startup catalog. Remaining extensions include:

1. **Active-row physics.** Give kernels device counts or worklists so the final partial chunk avoids simulating vacant rows.
2. **Packed persistent fields.** Replace maximum-width logical state with per-model spans when state padding itself becomes significant; preserve logical identity and policy routing when spans move.
3. **Sleeping in pooled storage.** Preserve or rebuild the derived caches skipped by sleeping kernels. The fixed-capacity GPU runner already supports sleeping.
4. **Tighter temporary reuse.** Reduce capture scratch through verified temporary lifetimes within a physics phase, beyond the current reuse across serial model branches.
5. **Runtime topology changes.** Editing bodies/joints inside an existing model requires rebuilding model-relative metadata, collision work and solver specializations. The current model catalog remains immutable.
6. **Richer tensor interfaces.** Provide model-specific observation/action views or packed offsets while retaining zero-copy ownership, epochs and device routing. A variable robot width cannot be represented in one unpadded rectangular tensor.

A Madrona-style host mapping service could later service growth during a long-running graph. That needs a defined device/host request protocol, memory publication ordering, failure handling and progress guarantees. The present `commit()` synchronizes the context and must not be called by a service while a kernel waits for it; that would deadlock. No GPU spin-wait allocator is included.

Sleeping remains useful within groups for temporarily inactive bodies. It is not a replacement for packed storage: the union model still has full `qpos`, `qvel`, body/geometry and other arrays, and the current compact solver allocates additional compact workspaces. VMM alone cannot remove those arrays or change fixed loop bounds. The prototype's measured gains come from preserving existing model-specialized physics while removing that unnecessary union.

## Reproduce and inspect

Raw JSON timings and final-state NPZ snapshots are in [`../../benchmarks/heterogeneous_results`](../../benchmarks/heterogeneous_results). The benchmark is [`../../benchmarks/heterogeneous.py`](../../benchmarks/heterogeneous.py). The sleep checkout is `../mjwarp-sleep-baseline`; it contains the PR applied to the same base as this fork.

```bash
# Run from mustafa_mjwarp, with the local Warp fork built and linked.
uv run --no-sync benchmarks/heterogeneous.py --mode sleep \
  --mjwarp-path ../mjwarp-sleep-baseline --nworld 8192 --nconmax 16 --njmax 64 \
  --output /tmp/sleep.json
uv run --no-sync benchmarks/heterogeneous.py --mode grouped --virtual \
  --mjwarp-path ../mjwarp-sleep-baseline --nworld 8192 --nconmax 16 --njmax 64 \
  --output /tmp/virtual.json
uv run --no-sync python -m warp.examples.core.example_dynamic_table --worlds 8192
uv run --no-sync pytest -n 8 mujoco_warp/experimental/*_test.py \
  mujoco_warp/_src/io_test.py mujoco_warp/_src/forward_test.py mujoco_warp/_src/sleep_test.py \
  mujoco_warp/_src/history_test.py
```

The complete experimental suite plus existing IO, forward, sleeping and history regressions completed with **383 passed and 11 skipped**. All repository pre-commit checks pass, including the kernel analyzer. The pooled and fixed-capacity GPU examples execute successfully.

Warp's focused tests are `warp/tests/test_virtual_memory.py`, `warp/tests/test_dynamic_table.py` and `warp/tests/test_capture_scratch.py` in the Warp fork. Six capture-scratch tests verify native scan, conditional branches, pointer reuse, graph/array/child-graph ownership, and bounded exhaustion. A full conditional mjwarp step with native scan and a nested solver loop also matched eager execution. They cover physical growth across the mapping granule under an existing graph, zero-copy DLPack ownership, bounded concurrent overflow, stable handles after grouping, deletion/reuse, batch overflow and component preservation. Existing Warp allocator/graph regression tests and a CUDA-disabled native syntax check also pass; the Warp HTML documentation builds without warnings.

The inspected Madrona checkout is `86457089d677d94e2826613c8ba62d180b50ab41`, and the Warp changes are based on `2832caa9014f7633759ef3976478e3928e741de5`. Changes in both development forks are intentionally left available in the working trees for inspection; no PR was published.

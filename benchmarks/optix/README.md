# OptiX batch rendering

[Other rendering benchmarks](OTHER_BENCHMARKS.md): primitives, cloth, and G1
heightfield, with side-by-side previews and measurements at their configured batch sizes.

**Latest results:** [ALOHA clutter vision on driver R595](CLUTTER.md), with four-camera
side-by-side renders and measurements through 2,048 worlds. The full batch renders
about 11.1× faster with OptiX. OptiX 9.1 now loads, but graph replay still fails;
the graph-capture investigation below describes the earlier R580 experiments.

Select the backend at context creation:

```python
rc = mjw.create_render_context(
    mjm, nworld=nworld, cam_res=(128, 128),
    render_rgb=True, render_depth=True, backend="optix",  # default: "warp"
)
mjw.refit_bvh(m, d, rc)
mjw.render(m, d, rc)
```

Both backends share camera settings, shading, materials, textures, shadows, supersampling, and RGB/depth/segmentation outputs.

## Setup

This experimental adapter supports Linux, Warp 1.15.x, and **OptiX SDK 9.0 headers**. It requires an OptiX-compatible NVIDIA GPU/driver, CUDA toolkit headers, and a C++17 compiler (`c++`). RT cores accelerate triangle intersection. Newer compatible drivers work with SDK 9.0; installing SDK 9.1 is not required. The adapter checks the header version explicitly.

```sh
git clone --depth 1 --branch v9.0.0 https://github.com/NVIDIA/optix-dev.git /path/to/optix
export OPTIX_PATH=/path/to/optix
export CUDA_HOME=/usr/local/cuda
uv sync --extra optix --extra dev
```

From the repository root, try the included mesh scene:

```sh
OPTIX_PATH=/path/to/optix uv run --extra optix python - <<'PY'
import mujoco
import mujoco_warp as mjw

mjm = mujoco.MjModel.from_xml_path("mujoco_warp/test_data/aloha_pot/scene.xml")
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=16)
rc = mjw.create_render_context(
    mjm, nworld=16, backend="optix", cam_active=[0], cam_res=(128, 128),
    render_rgb=True, render_depth=True, use_shadows=True,
)
mjw.refit_bvh(m, d, rc)
mjw.render(m, d, rc)
print("Rendered RGB buffer:", rc.rgb_data.numpy().shape)
PY
```

Change `backend="optix"` to `backend="warp"` for software traversal. After updating
simulation poses, call `refit_bvh` before `render`. Keep mesh-containing OptiX
rendering outside CUDA graph capture; see the limitations below.

The native bridge and device programs compile on first use and are cached beneath Warp's kernel cache. No system driver or installed Warp source is modified. The adapter currently uses private Warp code-generation APIs, so the optional dependency pins the tested 1.15 series.

## Architecture and refit cost

Meshes and heightfields have shared static OptiX geometry acceleration structures. One instance acceleration structure (IAS) holds mesh instances for the entire batch. Worlds are placed in a three-dimensional grid with spacing twice the model extent; rays receive the same translation. Visibility masks prune other worlds, and an any-hit program checks the complete world ID to prevent cross-world hits even when mask bits repeat or cells overlap. Instance IDs impose a checked 24-bit limit on `nworld * ngeom` for mesh scenes.

Analytic shapes and flexes retain shared Warp ray–primitive functions. Short analytic candidate lists avoid software traversal through mesh leaves; larger lists use the grouped Warp BVH. Gaussian splat processing remains shared software code. Scenes without triangle assets use the ordinary Warp launch path, including graph capture.

`refit_bvh` performs the grouped Warp refit and packs current mesh instances on the GPU. `render` then builds the single IAS before launching shading. **Reported render times include this hardware build; refit times alone do not represent all scene maintenance.** Scratch/output buffers are reused. BUILD supports changes to per-world `geom_dataid`; OptiX UPDATE does not allow arbitrary changes to child traversable handles. See NVIDIA's [handle restrictions](https://forums.developer.nvidia.com/t/updating-as-guarantees-optixtraversablehandle-unchanged/149366).

The original implementation submitted two native updates per world, or 256 calls at 128 worlds. Removing this submission overhead was much more effective than tuning individual updates. Warp cannot directly refit OptiX's opaque acceleration structures; it continues to refit the software hierarchy and prepare hardware instances.

Use the same Warp stream for simulation, refit, and rendering, or establish dependencies with CUDA events.

## CUDA graph capture and the upstream Warp addon

**Refit and instance packing support capture. Mesh-containing OptiX rendering currently rejects capture on the tested runtime.** Capture simulation/refit and render after replay, or use `backend="warp"` for a fully captured mesh pipeline. Primitive-only OptiX contexts use Warp launches and support full capture.

We inspected and built [nvtw/otk-pyoptix's Warp addon](https://github.com/nvtw/otk-pyoptix/tree/49e540d381d580f6aea1b1df81aef800d1cab77a/warp_optix), from `dev/add_dlss_support`. It uses Warp >=1.17 public addon registration, external-entry kernels, and AOT compilation. Its [viewer captures warmed launches with persistent parameter buffers](https://github.com/nvtw/otk-pyoptix/blob/49e540d381d580f6aea1b1df81aef800d1cab77a/warp_optix/warp_optix/pathtracing/pathtracing_viewer.py#L1591). This is a promising replacement for our private compiler adapter, but it has not been adopted as a project dependency.

The unmodified upstream package requires OptiX 9.1 and failed to import here with `Unsupported ABI version`. NVIDIA requires [R590 or newer for OptiX 9.1](https://developer.nvidia.com/designworks/optix/download); this machine has 580.159.03. In an isolated Warp 1.17 environment, rebuilding the same binding against OptiX 9.0 verified eager triangle rendering, but replay failed with CUDA error 700. Native bridge probes also failed: ordinary capture reported internal event-query errors, while relaxed capture produced missing output or invalid-resource errors.

[probe_optix_graph.py](../probe_optix_graph.py) preserves a minimal upstream-addon test that checks eager rendering followed by repeated graph replay with changing ray origins. Run it in a separate process with the optional upstream packages installed. A compatible R590+ runtime is needed to test the unmodified upstream graph path; upgrading alone has not been verified to make our dynamic batch pipeline capturable. No system driver changes were made.

## Measurements

Measured September 15, 2026: RTX PRO 6000 Blackwell Workstation Edition, driver 580.159.03, Warp 1.15.0, MuJoCo 3.12.1, OptiX 9.0. Each world renders one 128×128 camera with RGB, depth, and segmentation enabled. Seven repeats of ten eager calls use CUDA events, including host submission and GPU idle intervals. Compilation/context creation are excluded. Desktop activity was present and clocks were not locked. These compare eager execution on both backends, not captured Warp against eager OptiX.

Median milliseconds per ALOHA batch, including refit and hardware scene build:

| Worlds | Shadows | Warp frame | OptiX frame | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | No | 0.315 | 0.199 | 1.58× |
| 16 | No | 0.653 | 0.289 | 2.26× |
| 128 | No | 3.342 | 0.696 | 4.80× |
| 1 | Yes | 0.795 | 0.213 | 3.73× |
| 16 | Yes | 1.679 | 0.324 | 5.18× |
| 128 | Yes | 8.964 | 1.113 | 8.05× |
| 128, varied poses | Yes | 9.748 | 1.321 | 7.38× |

At 128 worlds, Warp refit costs about 0.030 ms; OptiX refit plus instance packing costs 0.039–0.040 ms. OptiX render, including the IAS build, costs 0.658 ms without shadows, 1.077 ms with shadows, and 1.284 ms with varied poses. Frame measurements are independent of the component measurements, so medians need not sum exactly.

The previous hybrid (software scene traversal, hardware mesh traversal) took 1.237/3.811 ms without/with shadows, and 9.036 ms with varied poses. The new hierarchy is approximately 6.84× faster than that hybrid on varied poses. The original per-world implementation took 5.688/5.946 ms without/with shadows.

Raw measurements include timing ranges, setup times, primitive-only cases, and image differences:

- Current: [results](results.json), [shadows](shadows.json), [varied poses](diverse_results.json).
- Previous hybrid: [results](hybrid_results.json), [shadows](hybrid_shadows.json), [varied poses](hybrid_diverse_results.json).
- Original per-world: [results](per_world_results.json), [shadows](per_world_shadows.json).
- Exploratory alternatives: [experiments](experiments.json).

### Other optimizations explored

For ALOHA at 128 worlds with shadows, smaller shadow payloads and register limits gave no material improvement over the previous hybrid's 3.81 ms. Interleaving rays across worlds slowed frames to 5.41–6.84 ms. A single hierarchy with overlapping worlds took 3.79 ms. Groups of eight worlds improved tracing but required enough builds to leave frame time at 2.88 ms. Encoding worlds as motion-transform keys took 1.33 ms with identical poses but 3.16 ms with varied poses, and complicated dynamic mesh selection. A traversal-only query also increased RGB error. These alternatives were not retained. Spatial packing with shared analytic intersection code gave the best retained full-frame results.

### Image differences

Outputs are **not pixel-identical**. At 128 static ALOHA worlds, segmentation differs at 274 of 2,097,152 pixels (0.0131%), versus 256 for the previous hybrid. Maximum depth difference is 0.4784 m and maximum RGB channel difference is 81. Grid translations introduce additional floating-point rounding and can change surface selection near boundaries; the large maximum error must not be interpreted as uniform depth noise.

With varied poses, segmentation differs at 362 pixels, versus 359 for the previous hybrid. Maximum depth difference is 1.9758 m for both implementations; maximum RGB difference is 240 for the new implementation and 242 for the hybrid. Large isolated differences therefore already existed before spatial packing. At 1 and 16 static worlds, segmentation differs at two pixels per world and maximum depth error is approximately 6.38 mm. These results do not establish pixel-perfect parity or accuracy at arbitrary coordinate scales.

## Reproduce and validate

```sh
OPTIX_PATH=/path/to/optix uv run benchmarks/render_backends.py \
  mujoco_warp/test_data/primitives.xml \
  mujoco_warp/test_data/aloha_pot/scene.xml \
  --worlds 1 16 128 --resolution 128 \
  --output benchmarks/optix/results.json
```

Add `--shadows` for shadow rays. Add `--pose-noise .2` to perturb hinge joints independently per world (Gaussian noise, seed 0, clamped to joint limits). Poses remain fixed during timing, but every frame executes refit, instance packing, and the hardware build; there is no unchanged-scene shortcut.

```sh
OPTIX_PATH=/path/to/optix uv run --extra optix --extra dev pytest \
  mujoco_warp/_src/render_test.py \
  mujoco_warp/_src/render_util_test.py \
  mujoco_warp/_src/render_optix_test.py -q
```

Validation: **140 passed**. Coverage includes existing rendering tests, per-world mesh selection, translated/rotated meshes, captured refit replay after motion, 17 worlds sharing visibility-mask bits, deforming flexes, heightfields, a nondefault stream, empty scenes, primitive render capture, and mesh render capture rejection that leaves CUDA usable. SDK-dependent tests skip when optional prerequisites are absent.

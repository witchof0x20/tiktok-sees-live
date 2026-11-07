# What TikTok sees, live

A local copy of [ayoships.com/tools/tiktok-face](https://ayoships.com/tools/tiktok-face) plus a live webcam
version that runs TikTok's two decoded face networks (`expnet`: attractiveness/mood, `agenet`: age/gender) on a
from-scratch WebGPU engine. Everything runs in the browser, on your device.

- `upstream/` – the original tool, mirrored with its original paths (`/upstream/tools/tiktok-face/`).
- `live/` – the live app (`/live/`) and the GPU engine (`live/gpu/`).
- `live/models/` – the face detector and landmark models converted from MediaPipe's `face_landmarker.task`.
- `live/test/` – bit-exactness check, face model and face finder checks, crop fidelity check, benchmarks, autotuner.
- `tools/` – TFLite converter, graph/bounds analysis, headless Chromium driver, dev server.

## Run

Camera frames are read with `MediaStreamTrackProcessor` and drawn with WebGPU rather than through a `<video>`
element, which shows blank frames on some Linux/Vulkan Chrome setups.

```sh
nix develop            # node, python, chromium
python3 tools/serve.py 8766   # static server with caching off
# open http://127.0.0.1:8766/live/
```

WebGPU on Linux Chrome needs `chrome://flags/#enable-unsafe-webgpu` (and Vulkan).

## The engine

The upstream reference engine (`engine.js`) emulates the nets' 16-bit fixed-point arithmetic in plain JS:
round-half-up rescales, int16 saturation, ReLU. The GPU engine reproduces it **bit for bit**: every output float
matches the reference exactly (`live/test/verify.html`). No quantization or approximation is introduced.

How it gets fast:

- **Exact integer math, proven safe.** Interval analysis over all weights shows every conv accumulator stays
  below 2^29.8 for *any* input, so i32 accumulation is exact (`tools/bounds.mjs`).
- **Full-rate 24-bit multiplies.** RDNA's 32-bit integer multiply is quarter rate. Operands are shifted into
  unsigned range (x+32768 < 2^16, w+2048 < 2^12) so the compiler can emit `v_mad_u32_u24`, then the offsets are
  removed algebraically (Σxw = Σx'w' − 2048Σx' − 32768Σw' + K·2^26, mod 2^32), with the weight term folded into
  the bias. Still bit-exact.
- **Fusion compiler** (`fuse.js`, `codegen.js`). Ops are grouped into kernels. Within a kernel, convs that feed
  spatial ops are staged in workgroup memory with halos; convs that only feed pointwise ops, plus all
  adds/concats, are recomputed in registers (concat is free, residual adds are epilogues). Global average pooling
  is fused as workgroup + global atomics. Kernels at the same depth, across branches *and* both nets, share one
  dispatch (horizontal fusion). Dead outputs (`predict_happy`) are pruned.
- **Register-blocked micro-kernels.** Each thread computes a Q-channel-group × P-pixel tile, so input loads are
  reused across output channels and weight loads across pixels. Depthwise rows slide over shared loads.
- **Two schedules.** `unit` spreads (channel, pixel) blocks over lanes. `px` makes channel loops
  workgroup-uniform so weights are fetched with scalar loads.
- **Autotuner** (`tune.js`, `live/test/tune.html`). Greedy fusion and per-kernel (tile, split, P, Q, schedule)
  choices driven by GPU timestamp measurements, saved to `live/tuned/<arch>.json` and loaded at runtime.
  Without a tuning file, a heuristic cost model is used.
- **Everything in one submit, no CPU round trips.** Per camera frame: upload → mips → face detector → NMS/ROIs →
  face landmarks (one batch item per face) → crop parameters → both TikTok nets (one batch item per face) →
  one readback. Face counts never come back to the CPU mid-frame: post-processing kernels write them to GPU
  buffers, and later stages size themselves with indirect dispatch. Crops are fused into each net's first conv,
  which samples the video texture directly. The small fc/softmax tail runs on the CPU with the reference's exact
  float code. Up to three frames are in flight.

## The face finder (`live/gpu/facefinder.js`)

A re-implementation of MediaPipe's FaceLandmarker on the same engine; MediaPipe itself is no longer in the loop.
`tools/convert_tflite.py` (with a small, bounds-checked flatbuffer reader in `tools/tflite.py`) converts the
BlazeFace short-range detector and the FaceMesh v2 landmark model out of `face_landmarker.task` into the engine's
float graph format offline, so the browser never parses TFLite. float16 weights are widened to f32 losslessly;
inference is f32, matching a CPU reference interpreter to ~1e-6 relative (`live/test/verify-face.html`).

The graph logic around the models is ported to two WGSL kernels:
- **detPost**: decodes the 896 SSD anchors, applies weighted NMS, removes the letterbox, builds rotated ROIs (eye
  keypoints, square long side, ×1.5) and merges them with ROIs tracked from the previous frame's landmarks.
- **lmPost**: presence score, projects the 478 landmarks back to the frame, computes next frame's tracking ROIs,
  and derives upstream's eye/mouth points and sanity filter plus the TikTok crop parameters.

Against MediaPipe on the same frames, landmarks agree to ~0.4–0.5% of the eye distance (`live/test/facefinder.html`).
The MediaPipe path is still available in the app (Face finder → MediaPipe workers) for comparison.

## Tuning

```sh
# stable clocks give stable timings (AMD):
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
node tools/cdp.mjs "http://127.0.0.1:8766/live/test/tune.html?k=16" "window.__tune && JSON.stringify(window.__tune)" 3000000 > tiktok.json
node tools/cdp.mjs "http://127.0.0.1:8766/live/test/tune.html?k=16&nets=face" "window.__tune && JSON.stringify(window.__tune)" 3000000 > face.json
# merge the two "nets" maps into live/tuned/<architecture>.json
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

# What TikTok sees, live

**[Open the live app →](https://witchof0x20.github.io/tiktok-sees-live/)**

A live webcam version of **[What TikTok sees](https://ayoships.com/tools/tiktok-face)** by
[Ayo](https://ayoships.com), who decoded the face networks TikTok runs on-device and built the original tool:
upload a photo, see the attractiveness score, age and gender guesses and mood TikTok's model assigns. All credit for
finding, decoding and explaining those networks goes to that project. This repo takes the same two networks
(`expnet`: attractiveness and mood, `agenet`: age and gender) and runs them on every camera frame. Not affiliated
with TikTok.

Everything runs in your browser, on your GPU. Nothing is uploaded.

With WebGPU (current Chrome/Edge on desktop and many Android devices, Safari 26+; on Linux Chrome enable
`chrome://flags/#enable-unsafe-webgpu` and Vulkan) everything runs on the GPU. Without it the app switches to a
**CPU version**: MediaPipe finds faces (on WebGL where available) and the TikTok nets run in a Rust/WebAssembly SIMD
engine, still bit-exact.

The app always **starts on the CPU version** (it's ready in about a second) and builds the WebGPU path in the
background: on first use on an unknown GPU that includes a one-time quick tune, then compiling kernels. It switches
over once that's done. If the WebGPU path fails on its first frames, it goes back to the CPU version instead of
showing a black screen. `?backend=cpu` or `?backend=gpu` pins one path.

## What's here

- `live/` – the app, and `live/gpu/`, the engine it runs on.
- `live/nets/` – the decoded TikTok network weights, as published by the original tool.
- `live/models/` – MediaPipe's face detector and landmark models, converted for the engine (from
  `live/vendor/face_landmarker.task`, Apache-2.0).
- `live/vendor/` – MediaPipe Tasks Vision (Apache-2.0), used only by the optional "MediaPipe workers" comparison mode.
- `live/tuned/` – autotuned kernel plans.
- `live/test/` – correctness checks, benchmarks and the autotuner.
- `tools/` – TFLite converter, graph/bounds analysis, headless Chromium driver, dev server.

## Run locally

```sh
nix develop                  # node, python, chromium, rust
./tools/build-wasm.sh        # the CPU engine -> live/cpu/tiktok_cpu.wasm
python3 tools/serve.py 8766  # static server with caching off
# open http://127.0.0.1:8766/live/
```

The original tool isn't redistributed here. `tools/fetch-upstream.sh` downloads it into `upstream/` (git-ignored)
so you can run it locally at `/upstream/tools/tiktok-face/`. Some checks (`verify.html`, `crop.html`,
`facefinder.html`, `bench-pipeline.html`) compare against its reference engine and sample photo and need it.
With it present, the app also gets a "Demo (sample photo)" source.

## How it works

Per camera frame, everything below is **one GPU submit with one readback**:

frame → texture + mips → face detector → NMS / face regions → face landmarks (one batch item per face) →
crop parameters → both TikTok nets (one batch item per face) → readback → preview and overlay.

Face counts never come back to the CPU mid-frame. Post-processing kernels write them to GPU buffers, and later
stages size themselves with indirect dispatch. Camera frames are read with `MediaStreamTrackProcessor` (a
`<video>` fallback where that's missing) and drawn with WebGPU.

### The TikTok nets: bit-exact

The original tool's engine emulates the nets' 16-bit fixed-point arithmetic in plain JS: round-half-up rescales,
int16 saturation, ReLU. The GPU engine reproduces it **bit for bit**: every output matches the reference exactly
(`live/test/verify.html`). No quantization or approximation is introduced.

- **Exact integer math, proven safe.** Interval analysis over all weights shows every conv accumulator stays below
  2^29.8 for *any* input, so i32 accumulation is exact (`tools/bounds.mjs`).
- **Full-rate 24-bit multiplies.** 32-bit integer multiply is quarter rate on RDNA. Operands are shifted into
  unsigned range (x+32768 < 2^16, w+2048 < 2^12) so the compiler emits `v_mad_u32_u24`, then the offsets are
  removed algebraically (Σxw = Σx'w' − 2048Σx' − 32768Σw' + K·2^26, mod 2^32), with the weight term folded into the
  bias.
- The crop (eye/mouth similarity transform, like the original's `drawCrop`) is fused into each net's first conv,
  which samples the video texture directly. Against the original's canvas crop, scores differ by at most 0.34
  (attractiveness) / 0.8 years (`live/test/crop.html`).

### The engine

A small compiler from op graphs to fused WGSL compute kernels (`graph.js`, `fuse.js`, `codegen.js`, `engine.js`):

- **Fusion.** Ops are grouped into kernels. Within a kernel, convs that feed spatial ops are staged in workgroup
  memory with halos, while pointwise ops and convs feeding only pointwise ops are recomputed in registers (concat is
  free, residual adds and activations are epilogues). Global average pooling is fused as workgroup + global atomics.
  Independent kernels at the same depth share one dispatch (horizontal fusion).
- **Register-blocked micro-kernels.** Each thread computes a Q-channel-group × P-pixel tile, and depthwise rows slide
  over shared loads. Two schedules: lanes over blocks, or workgroup-uniform channel loops so weights become scalar
  loads.
- **Fixed-point and float graphs.** NC4HW4 layout, int16 values or f32 bit patterns in the same buffers.
- **Autotuning** (`tune.js`). Fusion groups and per-kernel tile/split/block/schedule choices are made from GPU
  timestamp measurements. The full tune takes minutes and is bundled for AMD RDNA3 (`live/tuned/rdna-3.json`).
  On any other GPU, the app runs a **quick tune** once (a few seconds on a desktop GPU; skippable). It keeps the
  bundled plan's fusion groups, re-picks each group's parameters on your device, splits groups that don't fit
  your GPU, and caches the result in `localStorage`. Without a plan it uses one op per kernel.

On a Radeon 860M with one face, GPU time per frame is about 0.3 ms (face detector), 1.2 ms (landmarks) and 1.3 ms
(both TikTok nets): roughly 4.75 ms per frame, or about 200 frames per second of headroom. A camera usually
delivers 30–60. The original JS engine takes about 200 ms for the two TikTok nets alone.

### The CPU version (`wasm/`, `live/cpu/`)

For browsers without WebGPU. The TikTok nets run in a small Rust engine compiled to WebAssembly with SIMD128, one
worker per net:

- Same fixed-point semantics, **bit-identical** to the reference (`node tools/test-wasm.mjs`, which compares
  against the original engine from `tools/fetch-upstream.sh`).
- Channel-pair interleaved int16 tensors, so a SIMD load is 4 pixels × 2 channels, the exact operand shape of
  `i32x4.dot_i16x8_s`. 1×1 convs do 4 output channels × 8 pixels per step, and int16 saturation comes free from
  `i16x8.narrow_i32x4_s`. First convs use im2col dot products, and depthwise convs run on zero-bordered planes
  with widening multiplies.
- The program (shapes, ops, weights) is built in JS from the same parsed graph and fully validated in Rust before
  anything runs.
- On the dev laptop: 3.8 ms (expnet) and 9.0 ms (agenet) per face, against 72 ms and 150 ms for the reference JS.

Faces come from MediaPipe's FaceLandmarker in a worker (WebGL, or its CPU path), and the crops are the original's
canvas crop. GitHub Actions builds the WASM and deploys the site.

### The face finder

The original tool finds faces with MediaPipe's FaceLandmarker. Here it is re-implemented on the same engine
(`live/gpu/facefinder.js`):

- `tools/convert_tflite.py` (with a small, bounds-checked flatbuffer reader in `tools/tflite.py`) converts the
  BlazeFace short-range detector and FaceMesh v2 landmark model offline, so the browser never parses TFLite.
  float16 weights are widened to f32 losslessly, and inference matches a CPU reference to ~1e-6 relative
  (`live/test/verify-face.html`).
- **detPost** decodes the 896 SSD anchors, applies weighted NMS, removes the letterbox, builds rotated face regions
  (eye keypoints, square, ×1.5) and merges them with regions tracked from the previous frame's landmarks.
- **lmPost** computes presence, projects the 478 landmarks back to the frame, derives next frame's tracking regions,
  and computes the original tool's eye/mouth points and sanity filter plus the TikTok crop parameters.

Landmarks agree with MediaPipe's to ~0.4–0.5% of the eye distance (`live/test/facefinder.html`).

## Tuning a new GPU fully

```sh
# stable clocks give stable timings (AMD):
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
node tools/cdp.mjs "http://127.0.0.1:8766/live/test/tune.html?k=16" "window.__tune && JSON.stringify(window.__tune)" 3000000 > tiktok.json
node tools/cdp.mjs "http://127.0.0.1:8766/live/test/tune.html?k=16&nets=face" "window.__tune && JSON.stringify(window.__tune)" 3000000 > face.json
python3 tools/merge_tuned.py live/tuned/<architecture>.json tiktok.json face.json
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

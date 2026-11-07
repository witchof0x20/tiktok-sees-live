# What TikTok sees, live

A local copy of [ayoships.com/tools/tiktok-face](https://ayoships.com/tools/tiktok-face) plus a live webcam
version that runs TikTok's two decoded face networks (`expnet`: attractiveness/mood, `agenet`: age/gender) on a
from-scratch WebGPU engine. Everything runs in the browser, on your device.

- `upstream/` – the original tool, mirrored with its original paths (`/upstream/tools/tiktok-face/`).
- `live/` – the live app (`/live/`) and the GPU engine (`live/gpu/`).
- `live/test/` – bit-exactness check, crop fidelity check, benchmark, micro-benchmark, autotuner.

## Run

```sh
nix develop            # node, python, chromium
python3 -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/live/
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
- **Everything in one submit.** Frame copy → mips (only if a face is small) → crop (fused into the first conv
  of each net, sampling the video texture directly) → both nets for every face in the batch → readback of the
  pooled sums. The small fc/softmax tail runs on the CPU with the reference's exact float code. Up to three
  frames are in flight.

Face finding uses the same MediaPipe FaceLandmarker as upstream (video mode, GPU delegate), with the same
landmark picks and sanity filter.

## Tuning

```sh
# stable clocks give stable timings (AMD):
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
node tools/cdp.mjs "http://127.0.0.1:8765/live/test/tune.html?k=16" "window.__tune && JSON.stringify(window.__tune)" 3000000 > live/tuned/rdna-3.json
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

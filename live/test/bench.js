// Benchmarks: reference CPU engine vs fused GPU engine. Query: ?opt={...planner overrides}&batch=1&ref=0
import { requestDevice, createEngine } from '../gpu/engine.js';
import { FrameTexture, EXP, AGE, cropParams } from '../gpu/frame.js';
import { UP, NETS, loadReferenceEngine } from './upstream.js';

const q = new URLSearchParams(location.search);
const useTuned = !!q.get('tuned'), opt = JSON.parse(q.get('opt') || '{}'), batch = +(q.get('batch') || 1), runRef = q.get('ref') !== '0', iters = +(q.get('iters') || 200);
const lines = [];
const log = t => { lines.push(t); document.getElementById('log').textContent = lines.join('\n'); console.log(t); };
const WANT = { expnet: ['predict_attractive', 'prob_exp', 'predict_valence', 'predict_arousal'], agenet: ['predict_age', 'prob_gender'] };
const load = async name => { const [spec, buf] = await Promise.all([fetch(NETS + name + '.json').then(r => r.json()), fetch(NETS + name + '.bin').then(r => r.arrayBuffer())]); return { spec, buf, wanted: WANT[name] }; };
const noise = (S, seed) => { let s = seed; const o = new Int16Array(S * S * 3); for (let i = 0; i < o.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; o[i] = (s >>> 24) - 128; } return o; };
const median = a => { const b = [...a].sort((x, y) => x - y); return b[b.length >> 1]; };

(async () => {
  const out = {};
  try {
    const { device, info } = await requestDevice();
    const specs = await Promise.all([load('expnet'), load('agenet')]);
    log(`adapter ${info.vendor} ${info.architecture}; opt ${JSON.stringify(opt)}; batch ${batch}`);
    const inputs = Array.from({ length: batch }, (_, b) => [noise(256, b + 1), noise(224, b + 9)]);
    if (runRef && await loadReferenceEngine().then(() => true, () => (log('reference engine not available (tools/fetch-upstream.sh); skipping it'), false))) {
      const ref = specs.map(s => new self.FaceNet.Net(s.spec, s.buf));
      ref[0].run(inputs[0][0]); ref[1].run(inputs[0][1]);
      const t = [];
      for (let i = 0; i < 3; i++) { const t0 = performance.now(); ref[0].run(inputs[0][0]); ref[1].run(inputs[0][1]); t.push(performance.now() - t0); }
      out.refMs = median(t);
      log(`reference JS engine, both nets, 1 face: ${out.refMs.toFixed(1)} ms`);
    }
    const tuned = useTuned ? await fetch(`../tuned/${info.architecture}.json`).then(r => r.json()) : null;
    const eng = await createEngine(device, specs, { maxBatch: Math.max(batch, 1), source: 'tensor', opt, tuned });
    out.kernels = eng.kernelCount; out.dispatches = eng.dispatches; out.compileMs = eng.compileMs;
    log(`${eng.kernelCount} kernels / ${eng.dispatches} dispatches, compiled in ${eng.compileMs.toFixed(0)} ms`);
    // Live path: crop from a video-sized texture on the GPU (same as the webcam loop), unless ?path=tensor.
    let run = () => eng.runTensors(inputs);
    if (q.get('path') !== 'tensor') {
      eng.destroy();
      const live = await createEngine(device, specs, { maxBatch: Math.max(batch, 1), source: 'texture', opt, tuned });
      // Pixel content doesn't change the cost; a synthetic frame stands in for a camera frame.
      const frame = new FrameTexture(device), cv = Object.assign(document.createElement('canvas'), { width: 1280, height: 720 }), cx = cv.getContext('2d');
      const grad = cx.createLinearGradient(0, 0, 1280, 720); grad.addColorStop(0, '#c88'); grad.addColorStop(1, '#337'); cx.fillStyle = grad; cx.fillRect(0, 0, 1280, 720);
      const bmp = await createImageBitmap(cv);
      frame.upload(bmp, 1280, 720);
      const crops = Array.from({ length: batch }, (_, b) => { const e = [500 + 60 * b, 300], m = [505 + 60 * b, 420]; return [cropParams(EXP, 1280, 720, e, m), cropParams(AGE, 1280, 720, e, m)]; });
      run = () => live.runFrame(frame.view, crops);
      Object.assign(eng, { runTensors: run, destroy: () => live.destroy() });
      log('path: texture crop (live)');
    }
    for (let i = 0; i < 30; i++) await eng.runTensors(inputs);
    const lat = [];
    for (let i = 0; i < iters; i++) { const t0 = performance.now(); await eng.runTensors(inputs); lat.push(performance.now() - t0); }
    out.latMs = median(lat);
    log(`GPU latency (upload -> results on CPU): median ${out.latMs.toFixed(3)} ms, p10 ${[...lat].sort((a, b) => a - b)[iters / 10 | 0].toFixed(3)} ms`);
    // Throughput with the readback ring kept full.
    const t0 = performance.now(); let done = 0; const inflight = new Set();
    while (done < iters) {
      const p = eng.runTensors(inputs);
      if (!p) { await Promise.race(inflight); continue; }
      inflight.add(p); p.then(() => { inflight.delete(p); done++; });
    }
    await Promise.all(inflight);
    out.thrMs = (performance.now() - t0) / done;
    log(`GPU pipelined: ${out.thrMs.toFixed(3)} ms per batch (${(1000 / out.thrMs).toFixed(0)} batches/s)`);
    eng.destroy();

    const prof = await createEngine(device, specs, { maxBatch: Math.max(batch, 1), source: 'tensor', opt, profile: true, tuned });
    if (prof.canTime) {
      const acc = prof.stages.map(() => []);
      for (let i = 0; i < 60; i++) { const r = await prof.runTensors(inputs); r.times.forEach((t, j) => acc[j].push(t)); }
      const per = acc.map(median);
      out.gpuUs = per.reduce((a, b) => a + b, 0);
      log(`GPU timestamps: sum of per-dispatch medians ${out.gpuUs.toFixed(0)} us`);
      prof.stages.forEach((s, j) => {
        const macs = s.kernels.reduce((a, k) => a + k.group.plan.macs, 0) * batch;
        log(`  d${String(j).padStart(2)} ${per[j].toFixed(1).padStart(7)} us ${(macs / per[j] / 1e3).toFixed(1).padStart(6)} GMAC/s smem ${String(s.sharedBytes).padStart(5)}  ${s.kernels.map(k => `${k.net ? 'age' : 'exp'}:${k.group.members.size}op/T${k.group.plan.T}/S${k.group.plan.S}/P${k.group.plan.P}Q${k.group.plan.Q}/${k.group.plan.nwg}wg/${(k.group.plan.macs / 1e6).toFixed(2)}M`).join(' + ')}`);
      });
      out.per = per;
    }
    prof.destroy();
  } catch (err) { log(String(err.stack || err)); out.error = String(err); }
  window.__bench = out;
})();

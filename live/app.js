// Live loop. Frames come straight off the track (MediaStreamTrackProcessor), not through <video>.
//
// GPU mode (default), one submit and one readback per frame:
//   VideoFrame ─► texture ─► mips ─► face detector ─► NMS/ROIs ─► landmarks (indirect, per face) ─► crop params
//              ─► TikTok nets (indirect, per face) ─► readback;  the same texture feeds the WebGPU preview.
// MediaPipe mode (for comparison): MediaPipe FaceLandmarker in workers round-robin; each result is scored against the
// exact frame it came from (a pinned ring slot).
import { requestDevice, createEngine, createProgram, expScores, ageScores } from './gpu/engine.js';
import { runTail } from './gpu/graph.js';
import { FaceFinder, OUT_FLOATS } from './gpu/facefinder.js';
import { FrameTexture, EXP, AGE, cropParams, transform } from './gpu/frame.js';
import { Display } from './gpu/display.js';
import { Graphs } from './graphs.js';

const VENDOR = new URL('/upstream/tools/tiktok-face/vendor/', location.href).href;
const NETS = '/upstream/tools/tiktok-face/';
const MOOD = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const MIN_AGE = 16, MAX_FACES = 4;
const WANT = { expnet: ['predict_attractive', 'prob_exp', 'predict_valence', 'predict_arousal'], agenet: ['predict_age', 'prob_gender'] };

const $ = id => document.getElementById(id);
const view = $('view'), overlay = $('overlay'), ctx = overlay.getContext('2d'), stage = $('stage');
const say = t => { $('status').textContent = t; };
const graphs = new Graphs($('graphs'));

function Rate() { this.t = []; this.tick = now => { this.t.push(now); while (this.t.length && now - this.t[0] > 1000) this.t.shift(); }; this.value = () => this.t.length > 1 ? (this.t.length - 1) * 1000 / (this.t[this.t.length - 1] - this.t[0]) : 0; }
function Avg() { this.v = 0; this.n = 0; this.add = x => { this.n++; this.v += (x - this.v) / Math.min(this.n, 30); }; }
const stats = { cam: new Rate(), det: new Rate(), net: new Rate(), detMs: new Avg(), gpuMs: new Avg(), e2e: new Avg() };

let device, engine, display, ring = [], workers = [], mainFinder = null, stream = null, reader = null, running = false;
let finder, tiktok, gpuFrame, readRing = [], specs, tuned = null, adapterInfo;
let faces = [], facesId = -1, scored = [], scoredId = -1, frameId = 0, slotNext = 0, lastTs = -1;

// Same landmark picks and sanity filter as upstream face.js. pts is [x0, y0, x1, y1, ...] normalized.
function faceFrom(pts, w, h) {
  const P = i => [pts[2 * i] * w, pts[2 * i + 1] * h], mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const l = mid(P(33), P(133)), r = mid(P(362), P(263)), e = mid(l, r), m = mid(P(61), P(291));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pts.length; i += 2) { x0 = Math.min(x0, pts[i]); x1 = Math.max(x1, pts[i]); y0 = Math.min(y0, pts[i + 1]); y1 = Math.max(y1, pts[i + 1]); }
  return { e, m, eyes: Math.hypot(r[0] - l[0], r[1] - l[1]), span: Math.hypot(m[0] - e[0], m[1] - e[1]), box: [x0 * w, y0 * h, x1 * w, y1 * h], pts };
}
const believable = f => f.span >= 8 && f.eyes / f.span > 0.35 && f.eyes / f.span < 1.9;

async function loadSpec(name) {
  const [spec, buf] = await Promise.all([fetch(NETS + name + '.json').then(r => r.json()), fetch(NETS + name + '.bin').then(r => r.arrayBuffer())]);
  return { spec, buf, wanted: WANT[name] };
}

function startWorker() {
  return new Promise((resolve, reject) => {
    const w = new Worker('finder-worker.js');
    w.busy = false;
    w.onmessage = ({ data }) => {
      if (data.type === 'ready') resolve(w);
      else if (data.type === 'error') reject(new Error(data.message));
      else if (data.type === 'result') { w.busy = false; onLandmarks(data); }
    };
    w.onerror = e => reject(new Error(e.message || 'worker failed'));
    w.postMessage({ type: 'init', vendor: VENDOR, delegate: 'GPU', numFaces: MAX_FACES });
  });
}

async function setWorkers(n) {
  workers.forEach(w => w.terminate());
  workers = [];
  try { workers = await Promise.all(Array.from({ length: n }, startWorker)); }
  catch (err) {
    console.warn('finder workers unavailable, finding faces on the main thread', err);
    workers.forEach(w => w.terminate()); workers = [];
    if (!mainFinder) {
      const { FaceLandmarker } = await import(VENDOR + 'vision_bundle.js');
      mainFinder = await FaceLandmarker.createFromOptions({ wasmLoaderPath: VENDOR + 'vision_wasm_internal.js', wasmBinaryPath: VENDOR + 'vision_wasm_internal.wasm' }, {
        baseOptions: { modelAssetPath: VENDOR + 'face_landmarker.task', delegate: 'GPU' }, runningMode: 'VIDEO', numFaces: MAX_FACES,
        minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.6, minTrackingConfidence: 0.6 });
    }
  }
  // Each in-flight detection pins a ring slot; two spare slots keep the GPU from reading a slot being refilled.
  ring.forEach(f => f.tex?.destroy());
  ring = Array.from({ length: Math.max(1, workers.length) + 4 }, () => Object.assign(new FrameTexture(device), { pins: 0 }));
  $('finder').textContent = workers.length ? `${workers.length} worker${workers.length > 1 ? 's' : ''}` : 'main thread';
}

async function init() {
  say('Starting WebGPU…');
  ({ device, info: adapterInfo } = await requestDevice());
  device.lost.then(l => say(`GPU device lost: ${l.message}`));
  say('Loading nets…');
  specs = await Promise.all([loadSpec('expnet'), loadSpec('agenet')]);
  try { const r = await fetch(`tuned/${adapterInfo.architecture || 'default'}.json`); if (r.ok) tuned = await r.json(); } catch {}
  say(`Compiling fused kernels${tuned ? ' (tuned for ' + adapterInfo.architecture + ')' : ''}…`);
  finder = await FaceFinder.create(device, { base: 'models/', tuned });
  tiktok = await createProgram(device, specs, { maxBatch: MAX_FACES, source: 'texture', tuned, batchBuffer: finder.ttBatch, cropsBuffer: finder.ttCrops, label: 'tiktok' });
  gpuFrame = new FrameTexture(device);
  const bytes = OUT_FLOATS * 4 + tiktok.sumLen * 4;
  readRing = Array.from({ length: 3 }, () => ({ buf: device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));
  display = new Display(device, view);
  const k = finder.det.kernelCount + finder.lm.kernelCount + tiktok.kernelCount + 2, d = finder.det.dispatches + finder.lm.dispatches + tiktok.dispatches + 4;
  $('engine').textContent = `${adapterInfo.vendor} ${adapterInfo.architecture} · face detector + landmarks + both TikTok nets: ${k} kernels, ${d} dispatches per frame, one submit · ${tuned ? 'autotuned' : 'heuristic'} plan · TikTok nets bit-exact int16 fixed point`;
  await setMode();
}

async function setMode() {
  const mp = $('finderMode').value === 'mediapipe';
  $('workersLabel').hidden = !mp;
  if (!mp) { workers.forEach(w => w.terminate()); workers = []; $('finder').textContent = 'GPU, same submit'; finder.reset(); faces = []; scored = []; return; }
  say('Loading MediaPipe…');
  if (!engine) engine = await createEngine(device, specs, { maxBatch: MAX_FACES, source: 'texture', tuned });
  await setWorkers(+$('workers').value);
}

// Demo source: upstream's sample photo drifting and turning, as a 60 fps stream through the same path as a camera.
let demoTimer = 0;
async function demoStream(w, h) {
  const img = new Image(); img.src = NETS + 'sample.jpg'; await img.decode();
  const c = Object.assign(document.createElement('canvas'), { width: w, height: h }), x = c.getContext('2d');
  const t0 = performance.now();
  const paint = () => {
    const t = (performance.now() - t0) / 1000, s = h / 1024 * (0.9 + 0.25 * Math.sin(t * 0.7));
    x.setTransform(1, 0, 0, 1, 0, 0); x.fillStyle = '#3a3f4a'; x.fillRect(0, 0, w, h);
    x.translate(w / 2 + Math.sin(t * 0.9) * w * 0.15, h / 2 + Math.cos(t * 1.3) * h * 0.05); x.rotate(Math.sin(t * 0.6) * 0.25);
    x.drawImage(img, -512 * s, -512 * s, 1024 * s, 1024 * s);
  };
  clearInterval(demoTimer); paint(); demoTimer = setInterval(paint, 1000 / 60);
  return c.captureStream(60);
}

async function openCamera() {
  stream?.getTracks().forEach(t => t.stop());
  const [w, h] = $('res').value.split('x').map(Number);
  clearInterval(demoTimer);
  stream = $('src').value === 'demo' ? await demoStream(w, h)
    : await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 60 }, facingMode: 'user' } });
  faces = []; scored = [];
  const track = stream.getVideoTracks()[0];
  reader?.cancel().catch(() => {});
  const r = reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
  (async () => {
    for (;;) {
      const { value: frame, done } = await r.read().catch(() => ({ done: true }));
      if (done || r !== reader) { frame?.close(); break; }
      try { onFrame(frame); } catch (err) { console.error(err); } finally { frame.close(); }
    }
  })();
}

function resize(w, h) {
  if (overlay.width === w && overlay.height === h) return;
  overlay.width = view.width = w; overlay.height = view.height = h;
  stage.style.aspectRatio = `${w} / ${h}`;
}

const pending = new Map(); // frame id -> { slot, w, h, t }

function onFrame(frame) {
  if (!running) return;
  const now = performance.now();
  stats.cam.tick(now);
  const w = frame.displayWidth, h = frame.displayHeight;
  resize(w, h);
  if ($('finderMode').value === 'gpu') { onFrameGpu(frame, w, h, now); draw(w, h, frame); return; }
  const ts = Math.max(lastTs + 1, Math.round(frame.timestamp / 1000)); lastTs = ts;
  const id = ++frameId;
  // Ring slots stay pinned from capture until their nets are submitted; a frame with no free slot is dropped.
  let slot = -1;
  for (let k = 0; k < ring.length; k++) { const c = (slotNext + k) % ring.length; if (!ring[c].pins) { slot = c; break; } }
  if (slot < 0) return;
  slotNext = slot + 1;
  ring[slot].upload(frame, w, h);
  const enc = device.createCommandEncoder();
  display.draw(enc, ring[slot].view, $('mirror').checked);
  device.queue.submit([enc.finish()]);
  if (workers.length) {
    const wk = workers.find(x => !x.busy);
    if (wk) {
      wk.busy = true; ring[slot].pins++;
      pending.set(id, { slot, w, h, t: now });
      createImageBitmap(frame).then(bitmap => wk.postMessage({ type: 'frame', id, ts, bitmap }, [bitmap]), () => { wk.busy = false; ring[slot].pins--; pending.delete(id); });
    }
  } else if (mainFinder) {
    ring[slot].pins++;
    pending.set(id, { slot, w, h, t: now });
    const r = mainFinder.detectForVideo(frame, ts);
    onLandmarks({ id, ms: performance.now() - now, faces: r.faceLandmarks.map(L => Float32Array.from(L.flatMap(p => [p.x, p.y]))) });
  }
  draw(w, h, frame);
}

// The whole pipeline for one frame in one submit; results come back in one readback.
function onFrameGpu(frame, w, h, now) {
  gpuFrame.upload(frame, w, h);
  const enc = device.createCommandEncoder();
  const slot = readRing.find(r => !r.busy);
  if (slot) {
    slot.busy = true;
    const levels = gpuFrame.tex.mipLevelCount;
    finder.setFrame(w, h, levels, +$('faces').value);
    gpuFrame.mips(enc, levels - 1);
    finder.encode(enc, gpuFrame.view);
    tiktok.encode(enc, { view: gpuFrame.view });
    enc.copyBufferToBuffer(finder.out, 0, slot.buf, 0, OUT_FLOATS * 4);
    if (tiktok.sumLen) enc.copyBufferToBuffer(tiktok.sums, 0, slot.buf, OUT_FLOATS * 4, tiktok.sumLen * 4);
  }
  display.draw(enc, gpuFrame.view, $('mirror').checked);
  device.queue.submit([enc.finish()]);
  if (!slot) return;
  const t1 = performance.now();
  slot.buf.mapAsync(GPUMapMode.READ).then(() => {
    const all = slot.buf.getMappedRange();
    const out = new Float32Array(all.slice(0, OUT_FLOATS * 4)), sums = new Int32Array(all.slice(OUT_FLOATS * 4));
    slot.buf.unmap(); slot.busy = false;
    const t = performance.now();
    stats.gpuMs.add(t - t1); stats.e2e.add(t - now); stats.det.tick(t); stats.net.tick(t);
    const found = FaceFinder.decode(out).filter(f => f.presence >= finder.minPresence).map(f => {
      const pts = new Float32Array(f.pts.length);
      for (let i = 0; i < pts.length; i += 2) { pts[i] = f.pts[i] / w; pts[i + 1] = f.pts[i + 1] / h; }
      return { ...f, pts, span: Math.hypot(f.m[0] - f.e[0], f.m[1] - f.e[1]) };
    }).sort((a, b) => b.span - a.span);
    faces = found;
    scored = found.filter(f => f.slot >= 0).map(f => ({ ...expScores(runTail(tiktok.nets[0], specs[0].buf, sums, f.slot)), ...ageScores(runTail(tiktok.nets[1], specs[1].buf, sums, f.slot)), e: f.e, m: f.m }));
    graphs.push(t, scored.find(s => s.e === found[0]?.e) ?? null);
  }, () => { slot.busy = false; });
}

function onLandmarks({ id, faces: raw, ms }) {
  const p = pending.get(id);
  pending.delete(id);
  if (!p) return;
  const fr = ring[p.slot];
  try { score(id, p, fr, raw, ms); } finally { fr.pins--; }
}

function score(id, p, fr, raw, ms) {
  stats.det.tick(performance.now()); stats.detMs.add(ms);
  if (id < facesId) return; // a later frame already reported
  const found = raw.map(a => faceFrom(a, p.w, p.h)).filter(believable).sort((a, b) => b.span - a.span).slice(0, +$('faces').value);
  faces = found; facesId = id;
  if (!found.length) graphs.push(performance.now(), null);
  if (!found.length || !engine.idle()) return;
  const crops = found.map(f => [cropParams(EXP, p.w, p.h, f.e, f.m), cropParams(AGE, p.w, p.h, f.e, f.m)]);
  const lvl = Math.max(...crops.flat().map(c => c.lvl));
  const sent = found.map(f => ({ e: f.e, m: f.m })), t1 = performance.now();
  engine.runFrame(fr.view, crops, enc => fr.mips(enc, lvl))?.then(({ res }) => {
    const t = performance.now();
    stats.gpuMs.add(t - t1); stats.e2e.add(t - p.t); stats.net.tick(t);
    if (id < scoredId) return;
    scoredId = id;
    scored = res.map((r, i) => ({ ...expScores(r[0]), ...ageScores(r[1]), e: sent[i].e, m: sent[i].m }));
    graphs.push(t, scored[0]); // faces are sorted largest first
  });
}

// Pair each visible face with the nearest scored face from the latest finished submit.
function match(f) {
  let best = null, bd = Infinity;
  for (const s of scored) { const d = Math.hypot(s.e[0] - f.e[0], s.e[1] - f.e[1]); if (d < bd) { bd = d; best = s; } }
  return best && bd < f.span ? best : null;
}

function draw(w, h, src) {
  const mirror = $('mirror').checked, X = x => mirror ? w - x : x;
  ctx.clearRect(0, 0, w, h);
  const k = w / 1280;
  faces.forEach((f, i) => {
    if ($('mesh').checked) {
      ctx.fillStyle = 'rgba(37,244,238,.6)';
      for (let j = 0; j < f.pts.length; j += 4) ctx.fillRect(X(f.pts[j] * w) - 1.5 * k, f.pts[j + 1] * h - 1.5 * k, 3 * k, 3 * k);
    }
    const [x0, y0, x1, y1] = f.box, bx = Math.min(X(x0), X(x1)), bw = Math.abs(x1 - x0);
    ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 2 * k;
    ctx.beginPath(); ctx.roundRect(bx, y0, bw, y1 - y0, 10 * k); ctx.stroke();
    const s = match(f), lines = [];
    if (s) {
      const ok = s.age >= MIN_AGE, male = s.boy >= 0.5, top = s.exp.map((p, j) => [MOOD[j], p]).sort((a, b) => b[1] - a[1])[0];
      lines.push([ok ? s.attractive.toFixed(1) : '–', ok ? 'Attractiveness' : 'No score under 16']);
      lines.push([`${Math.round(s.age)}`, 'Age guess']);
      lines.push([`${male ? 'Male' : 'Female'} ${Math.round(100 * (male ? s.boy : 1 - s.boy))}%`, 'Gender guess']);
      lines.push([`${top[0]} ${Math.round(top[1] * 100)}%`, 'Mood']);
    } else lines.push(['…', 'Scoring']);
    drawCard(bx, y1 + 8 * k, Math.max(bw, 190 * k), lines, k);
    if ($('crops').checked) drawCrops(f, i, w, h, k, src);
  });
}

function drawCard(x, y, W, lines, k) {
  const pad = 10 * k, lh = 26 * k, H = pad * 2 + lines.length * lh;
  if (y + H > overlay.height) y = overlay.height - H - 4;
  ctx.fillStyle = 'rgba(18,18,17,.78)';
  ctx.beginPath(); ctx.roundRect(x, y, W, H, 10 * k); ctx.fill();
  lines.forEach(([big, small], i) => {
    const yy = y + pad + (i + 0.75) * lh;
    ctx.fillStyle = i === 0 ? '#fe2c55' : '#f1efe9';
    ctx.font = `${i === 0 ? 700 : 600} ${(i === 0 ? 22 : 17) * k}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'left'; ctx.fillText(big, x + pad, yy);
    ctx.fillStyle = '#9a958b'; ctx.font = `${12 * k}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'right'; ctx.fillText(small, x + W - pad, yy);
  });
  ctx.textAlign = 'left';
}

// What the nets are fed: the same similarity crop, drawn with canvas here just for display.
function drawCrops(f, i, w, h, k, src) {
  const S = 96 * k, y = 8 * k + i * (S + 8 * k);
  [EXP, AGE].forEach((t, j) => {
    const x = 8 * k + j * (S + 8 * k), g = transform(t, f.e, f.m), sc = S / t.S;
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, S, S); ctx.clip();
    ctx.fillStyle = '#000'; ctx.fillRect(x, y, S, S);
    ctx.setTransform(g.re * sc, g.im * sc, -g.im * sc, g.re * sc, x + g.tx * sc, y + g.ty * sc);
    ctx.drawImage(src, 0, 0, w, h);
    ctx.restore();
    ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, S - 1, S - 1);
  });
}

function hud() {
  if (!running) return;
  $('fpsCam').textContent = `${stats.cam.value().toFixed(0)} fps`;
  $('fpsDet').textContent = `${stats.det.value().toFixed(0)} fps`;
  $('msDet').textContent = $('finderMode').value === 'gpu' ? 'in the submit' : `${stats.detMs.v.toFixed(1)} ms`;
  $('fpsNet').textContent = `${stats.net.value().toFixed(0)} fps`;
  $('msNet').textContent = `${stats.gpuMs.v.toFixed(1)} ms`;
  $('msE2e').textContent = `${stats.e2e.v.toFixed(1)} ms`;
  setTimeout(hud, 250);
}

async function start() {
  $('start').disabled = true;
  try {
    if (!engine) await init();
    say('Opening camera…');
    await openCamera();
    stage.dataset.state = 'live'; $('hud').hidden = false; $('graphs').hidden = !$('showGraphs').checked;
    running = true; hud();
  } catch (err) {
    console.error(err);
    say(String(err.message || err));
    $('start').disabled = false;
  }
}

$('start').addEventListener('click', start);
for (const id of ['res', 'src']) $(id).addEventListener('change', () => { if (running) openCamera().catch(err => say(String(err))); });
$('workers').addEventListener('change', () => { if (engine && $('finderMode').value === 'mediapipe') setWorkers(+$('workers').value); });
$('finderMode').addEventListener('change', () => { if (finder) setMode().catch(err => say(String(err))); });
$('showGraphs').addEventListener('change', () => { if (running) $('graphs').hidden = !$('showGraphs').checked; });
// ?src=demo&workers=2&res=1280x720&faces=1&autostart=1 (handy for benchmarking)
const qp = new URLSearchParams(location.search);
for (const k of ['src', 'workers', 'res', 'faces', 'finderMode']) if (qp.has(k)) $(k).value = qp.get(k);
if (qp.has('autostart') && $('src').value === 'demo') start();
if (!navigator.gpu) say('This needs WebGPU. On Linux Chrome, enable chrome://flags/#enable-unsafe-webgpu (and Vulkan).');
window.__live = { stats, get device() { return device; }, get display() { return display; }, get ring() { return ring; }, get faces() { return faces; }, get scored() { return scored; }, get engine() { return engine; }, get workers() { return workers.length; } };

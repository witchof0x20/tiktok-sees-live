// Live loop: webcam frame -> MediaPipe face landmarks -> GPU crop + both fused nets in one submit -> overlay.
// Detection runs every camera frame; scoring is pipelined (up to three submits in flight) and never blocks drawing.
import { requestDevice, createEngine, expScores, ageScores } from './gpu/engine.js';
import { FrameTexture, EXP, AGE, cropParams, transform } from './gpu/frame.js';

const VENDOR = '/upstream/tools/tiktok-face/vendor/';
const NETS = '/upstream/tools/tiktok-face/';
const MOOD = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const MIN_AGE = 16, MAX_FACES = 4;
const WANT = { expnet: ['predict_attractive', 'prob_exp', 'predict_valence', 'predict_arousal'], agenet: ['predict_age', 'prob_gender'] };

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), ctx = overlay.getContext('2d'), stage = $('stage');
const say = t => { $('status').textContent = t; };

let device, engine, frame, finder, stream = null;
let faces = [], scored = [], running = false;
const stats = { cam: new Rate(), net: new Rate(), det: new Avg(), gpu: new Avg() };

function Rate() { this.t = []; this.tick = now => { this.t.push(now); while (this.t.length && now - this.t[0] > 1000) this.t.shift(); }; this.value = () => this.t.length > 1 ? (this.t.length - 1) * 1000 / (this.t[this.t.length - 1] - this.t[0]) : 0; }
function Avg() { this.v = 0; this.n = 0; this.add = x => { this.n++; this.v += (x - this.v) / Math.min(this.n, 30); }; }

// Same landmark picks and sanity filter as upstream face.js.
function faceFrom(L, w, h) {
  const P = i => [L[i].x * w, L[i].y * h], mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const l = mid(P(33), P(133)), r = mid(P(362), P(263)), e = mid(l, r), m = mid(P(61), P(291));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of L) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  return { e, m, eyes: Math.hypot(r[0] - l[0], r[1] - l[1]), span: Math.hypot(m[0] - e[0], m[1] - e[1]), box: [x0 * w, y0 * h, x1 * w, y1 * h], L };
}
const believable = f => f.span >= 8 && f.eyes / f.span > 0.35 && f.eyes / f.span < 1.9;

async function loadSpec(name) {
  const [spec, buf] = await Promise.all([fetch(NETS + name + '.json').then(r => r.json()), fetch(NETS + name + '.bin').then(r => r.arrayBuffer())]);
  return { spec, buf, wanted: WANT[name] };
}

async function init() {
  say('Starting WebGPU…');
  let info;
  ({ device, info } = await requestDevice());
  device.lost.then(l => say(`GPU device lost: ${l.message}`));
  say('Loading nets…');
  const specs = await Promise.all([loadSpec('expnet'), loadSpec('agenet')]);
  let tuned = null;
  try { const r = await fetch(`tuned/${info.architecture || 'default'}.json`); if (r.ok) tuned = await r.json(); } catch {}
  say(`Compiling fused kernels${tuned ? ' (tuned for ' + info.architecture + ')' : ''}…`);
  engine = await createEngine(device, specs, { maxBatch: MAX_FACES, source: 'texture', tuned });
  frame = new FrameTexture(device);
  $('engine').textContent = `${info.vendor} ${info.architecture} · ${engine.kernelCount} fused kernels in ${engine.dispatches} dispatches · ${tuned ? 'autotuned' : 'heuristic'} plan · bit-exact int16 fixed point`;
  say('Loading face finder…');
  const { FaceLandmarker } = await import(VENDOR + 'vision_bundle.js');
  const fileset = { wasmLoaderPath: VENDOR + 'vision_wasm_internal.js', wasmBinaryPath: VENDOR + 'vision_wasm_internal.wasm' };
  const make = delegate => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: VENDOR + 'face_landmarker.task', delegate }, runningMode: 'VIDEO', numFaces: MAX_FACES,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.6, minTrackingConfidence: 0.6,
  });
  try { finder = await make('GPU'); } catch (err) { console.warn('GPU face finder failed, using CPU', err); finder = await make('CPU'); }
}

async function openCamera() {
  stream?.getTracks().forEach(t => t.stop());
  const [w, h] = $('res').value.split('x').map(Number);
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 60 }, facingMode: 'user' } });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
}

let lastTs = -1;
function onFrame(now, meta) {
  if (!running) return;
  video.requestVideoFrameCallback(onFrame);
  stats.cam.tick(now);
  const w = video.videoWidth, h = video.videoHeight;
  const ts = Math.max(lastTs + 1, Math.round(meta.mediaTime * 1000)); lastTs = ts;
  const t0 = performance.now();
  const res = finder.detectForVideo(video, ts);
  stats.det.add(performance.now() - t0);
  const limit = +$('faces').value;
  faces = res.faceLandmarks.map(L => faceFrom(L, w, h)).filter(believable).sort((a, b) => b.span - a.span).slice(0, limit);

  if (faces.length && engine.idle()) {
    frame.upload(video, w, h);
    const crops = faces.map(f => [cropParams(EXP, w, h, f.e, f.m), cropParams(AGE, w, h, f.e, f.m)]);
    const lvl = Math.max(...crops.flat().map(c => c.lvl));
    const sent = faces.map(f => ({ e: f.e, m: f.m })), t1 = performance.now();
    engine.runFrame(frame.view, crops, enc => frame.mips(enc, lvl))?.then(({ res }) => {
      stats.gpu.add(performance.now() - t1);
      stats.net.tick(performance.now());
      scored = res.map((r, i) => ({ ...expScores(r[0]), ...ageScores(r[1]), e: sent[i].e, m: sent[i].m }));
    });
  }
  draw(w, h);
}

// Pair each visible face with the nearest scored face from the latest finished submit.
function match(f) {
  let best = null, bd = Infinity;
  for (const s of scored) { const d = Math.hypot(s.e[0] - f.e[0], s.e[1] - f.e[1]); if (d < bd) { bd = d; best = s; } }
  return best && bd < f.span ? best : null;
}

function draw(w, h) {
  const mirror = $('mirror').checked, X = x => mirror ? w - x : x;
  ctx.clearRect(0, 0, w, h);
  const k = w / 1280;
  faces.forEach((f, i) => {
    if ($('mesh').checked) {
      ctx.fillStyle = 'rgba(37,244,238,.55)';
      for (let j = 0; j < f.L.length; j += 2) ctx.fillRect(X(f.L[j].x * w) - k, f.L[j].y * h - k, 2 * k, 2 * k);
    }
    const [x0, y0, x1, y1] = f.box, bx = Math.min(X(x0), X(x1)), bw = Math.abs(x1 - x0);
    ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 2 * k;
    ctx.beginPath(); ctx.roundRect(bx, y0, bw, y1 - y0, 10 * k); ctx.stroke();
    const s = match(f);
    const lines = [];
    if (s) {
      const ok = s.age >= MIN_AGE, male = s.boy >= 0.5, top = s.exp.map((p, j) => [MOOD[j], p]).sort((a, b) => b[1] - a[1])[0];
      lines.push([ok ? s.attractive.toFixed(1) : '–', ok ? 'Attractiveness' : 'No score under 16']);
      lines.push([`${Math.round(s.age)}`, 'Age guess']);
      lines.push([`${male ? 'Male' : 'Female'} ${Math.round(100 * (male ? s.boy : 1 - s.boy))}%`, 'Gender guess']);
      lines.push([`${top[0]} ${Math.round(top[1] * 100)}%`, 'Mood']);
    } else lines.push(['…', 'Scoring']);
    drawCard(bx, y1 + 8 * k, Math.max(bw, 190 * k), lines, k);
    if ($('crops').checked) drawCrops(f, i, w, h, k);
  });
}

function drawCard(x, y, wmin, lines, k) {
  const pad = 10 * k, lh = 26 * k, W = wmin, H = pad * 2 + lines.length * lh;
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
function drawCrops(f, i, w, h, k) {
  const S = 96 * k, y = 8 * k + i * (S + 8 * k);
  [EXP, AGE].forEach((t, j) => {
    const x = 8 * k + j * (S + 8 * k), g = transform(t, f.e, f.m), sc = S / t.S;
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, S, S); ctx.clip();
    ctx.fillStyle = '#000'; ctx.fillRect(x, y, S, S);
    ctx.setTransform(g.re * sc, g.im * sc, -g.im * sc, g.re * sc, x + g.tx * sc, y + g.ty * sc);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();
    ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, S - 1, S - 1);
  });
}

function hud() {
  if (!running) return;
  $('fpsCam').textContent = `${stats.cam.value().toFixed(0)} fps`;
  $('msDet').textContent = `${stats.det.v.toFixed(1)} ms`;
  $('msNet').textContent = `${stats.gpu.v.toFixed(1)} ms`;
  $('fpsNet').textContent = `${stats.net.value().toFixed(0)} /s`;
  setTimeout(hud, 250);
}

async function start() {
  $('start').disabled = true;
  try {
    if (!engine) await init();
    say('Opening camera…');
    await openCamera();
    stage.dataset.state = 'live'; $('hud').hidden = false;
    running = true; video.requestVideoFrameCallback(onFrame); hud();
  } catch (err) {
    console.error(err);
    say(String(err.message || err));
    $('start').disabled = false;
  }
}

$('start').addEventListener('click', start);
$('res').addEventListener('change', () => { if (running) openCamera().catch(err => say(String(err))); });
$('mirror').addEventListener('change', () => { stage.dataset.mirror = $('mirror').checked ? '1' : '0'; });
stage.dataset.mirror = '1';
if (!navigator.gpu) say('This needs WebGPU. On Linux Chrome, enable chrome://flags/#enable-unsafe-webgpu (and Vulkan).');
window.__live = { stats, get faces() { return faces; }, get scored() { return scored; }, get engine() { return engine; } };

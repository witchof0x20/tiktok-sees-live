// GPU face finder: a re-implementation of MediaPipe's FaceLandmarker (BlazeFace short-range detector + FaceMesh v2
// landmarks + ROI tracking) on the fused engine, encoded into the same submit as everything else:
//
//   frame texture ─► detector (letterboxed 128x128) ─► detPost: anchors, weighted NMS, ROIs, merge with tracked ROIs
//                 ─► landmarks (256x256 per ROI, batch = GPU count, indirect) ─► lmPost: presence, projection,
//                    next-frame ROIs (tracking), eye/mouth points, crop params + count for the TikTok nets.
//
// No CPU round trip inside a frame: counts and crop parameters stay on the GPU and size later dispatches indirectly.
import { createProgram, writeCrops } from './engine.js';
import { EXP, AGE } from './frame.js';

export const MAX_FACES = 4;
export const N_LM = 478;
export const FACE_STRIDE = 16 + 2 * N_LM; // floats per face in the output block
const OUT_FLOATS = 4 + MAX_FACES * FACE_STRIDE;

const MODELS = { face_detector: ['t160', 't163', 't166', 't169'], face_landmarks: ['t472', 't473'] };
async function loadModel(base, name) {
  const [ir, buf] = await Promise.all([fetch(`${base}${name}.json`).then(r => r.json()), fetch(`${base}${name}.bin`).then(r => r.arrayBuffer())]);
  return { ir, buf, wanted: MODELS[name], dtype: 'f32' };
}

const COMMON = `struct Crop { m: vec4<f32>, t: vec2<f32>, lvl: i32, pad: i32 }
struct Params { frame: vec2<f32>, detScale: f32, minDet: f32, detOff: vec2<f32>, minPresence: f32, nmsIou: f32, trackIou: f32, maxFaces: i32, levels: i32, pad: i32 }
struct Track { count: i32, pad0: i32, pad1: i32, pad2: i32, rects: array<vec4<f32>, ${MAX_FACES}> }
const PI = 3.14159265358979;
fn normAngle(a: f32) -> f32 { return a - 2.0 * PI * floor((a + PI) / (2.0 * PI)); }
fn levelSize(lvl: i32) -> vec2<f32> {
  var s = vec2<i32>(params.frame);
  for (var i = 0; i < lvl; i++) { s = max(vec2<i32>(1), s / 2); }
  return vec2<f32>(s);
}
// Crop whose pixel centers d map to frame point c + R(rot) * (d / size - 0.5) * side, sampled from a mip level.
fn roiCrop(r: vec4<f32>, size: f32) -> Crop {
  var lvl = 0; var sc = size / r.z;
  loop { if (sc >= 0.25 || lvl >= params.levels - 1) { break; } sc *= 2.0; lvl++; }
  let k = r.z / size, c = cos(r.w), s = sin(r.w), ls = levelSize(lvl) / params.frame;
  var o: Crop;
  o.m = vec4<f32>(k * c * ls.x, k * s * ls.y, -k * s * ls.x, k * c * ls.y);
  o.t = vec2<f32>(r.x - (c - s) * r.z * 0.5, r.y - (s + c) * r.z * 0.5) * ls;
  o.lvl = lvl; o.pad = 0;
  return o;
}
fn rectIou(a: vec4<f32>, b: vec4<f32>) -> f32 {
  let ia = max(vec2<f32>(0.0), min(a.xy + a.z * 0.5, b.xy + b.z * 0.5) - max(a.xy - a.z * 0.5, b.xy - b.z * 0.5));
  let i = ia.x * ia.y;
  return i / (a.z * a.z + b.z * b.z - i);
}`;

function detPostSource(o) {
  const h = (name, n) => `fn ${name}(c: i32, y: i32, x: i32) -> f32 { return bitcast<vec4<f32>>(det[${n.off} + ((c / 4) * ${n.H} + y) * ${n.W} + x])[c % 4]; }`;
  return `${COMMON}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> det: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> track: Track;
@group(0) @binding(3) var<storage, read_write> rois: array<vec4<f32>, ${MAX_FACES}>;
@group(0) @binding(4) var<storage, read_write> lmBatch: vec4<i32>;
@group(0) @binding(5) var<storage, read_write> lmCrops: array<Crop, ${MAX_FACES}>;
${h('cls16', o.cls16)}
${h('cls8', o.cls8)}
${h('reg16', o.reg16)}
${h('reg8', o.reg8)}
const MAXC = 128;
var<workgroup> cnt: atomic<i32>;
var<workgroup> cScore: array<f32, MAXC>;
var<workgroup> cBox: array<vec4<f32>, MAXC>;     // xmin, ymin, xmax, ymax (frame px)
var<workgroup> cKp: array<vec4<f32>, MAXC>;      // keypoints 0 and 1 (eyes), frame px
var<workgroup> used: array<bool, MAXC>;

fn toFrame(u: vec2<f32>) -> vec2<f32> { return u * params.detScale + params.detOff; }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let t = i32(li);
  if (t == 0) { atomicStore(&cnt, 0); }
  workgroupBarrier();
  for (var a = t; a < 896; a += 256) {
    var logit: f32; var anc: vec2<f32>; var r: array<f32, 8>;
    if (a < 512) {
      let cell = a / 2; let k = a % 2; let y = cell / 16; let x = cell % 16;
      logit = cls16(k, y, x); anc = (vec2<f32>(f32(x), f32(y)) + 0.5) / 16.0;
      for (var j = 0; j < 8; j++) { r[j] = reg16(k * 16 + j, y, x); }
    } else {
      let b = a - 512; let cell = b / 6; let k = b % 6; let y = cell / 8; let x = cell % 8;
      logit = cls8(k, y, x); anc = (vec2<f32>(f32(x), f32(y)) + 0.5) / 8.0;
      for (var j = 0; j < 8; j++) { r[j] = reg8(k * 16 + j, y, x); }
    }
    let score = 1.0 / (1.0 + exp(-clamp(logit, -100.0, 100.0)));
    if (score < params.minDet) { continue; }
    let i = atomicAdd(&cnt, 1);
    if (i >= MAXC) { continue; }
    let c = vec2<f32>(r[0], r[1]) / 128.0 + anc; let wh = vec2<f32>(r[2], r[3]) / 128.0;
    let lo = toFrame(c - wh * 0.5); let hi = toFrame(c + wh * 0.5);
    cScore[i] = score; cBox[i] = vec4<f32>(lo, hi);
    cKp[i] = vec4<f32>(toFrame(vec2<f32>(r[4], r[5]) / 128.0 + anc), toFrame(vec2<f32>(r[6], r[7]) / 128.0 + anc));
    used[i] = false;
  }
  workgroupBarrier();
  if (t != 0) { return; }
  let n = min(atomicLoad(&cnt), MAXC);
  // Start from the tracked faces (deduplicated), then add new detections that don't overlap them.
  var out = 0;
  for (var i = 0; i < track.count && out < params.maxFaces; i++) {
    let r = track.rects[i];
    var dup = false;
    for (var j = 0; j < out; j++) { if (rectIou(rois[j], r) > params.trackIou) { dup = true; } }
    if (!dup) { rois[out] = r; out++; }
  }
  // Weighted NMS: repeatedly take the best remaining box, average every box overlapping it by score.
  loop {
    var best = -1; var bs = -1.0;
    for (var i = 0; i < n; i++) { if (!used[i] && cScore[i] > bs) { bs = cScore[i]; best = i; } }
    if (best < 0 || out >= params.maxFaces) { break; }
    let bb = cBox[best];
    var wsum = 0.0; var box = vec4<f32>(0.0); var kp = vec4<f32>(0.0);
    for (var i = 0; i < n; i++) {
      if (used[i]) { continue; }
      let o = cBox[i];
      let iw = max(0.0, min(bb.z, o.z) - max(bb.x, o.x)); let ih = max(0.0, min(bb.w, o.w) - max(bb.y, o.y));
      let inter = iw * ih; let uni = (bb.z - bb.x) * (bb.w - bb.y) + (o.z - o.x) * (o.w - o.y) - inter;
      if (uni > 0.0 && inter / uni > params.nmsIou) { used[i] = true; wsum += cScore[i]; box += cScore[i] * o; kp += cScore[i] * cKp[i]; }
    }
    box /= wsum; kp /= wsum;
    // DetectionsToRects (rotation from right eye to left eye) + RectTransformation (square long side, x1.5).
    let size = box.zw - box.xy;
    let rot = normAngle(-atan2(-(kp.w - kp.y), kp.z - kp.x));
    let rect = vec4<f32>((box.xy + box.zw) * 0.5, max(size.x, size.y) * 1.5, rot);
    var dup = false;
    for (var j = 0; j < out; j++) { if (rectIou(rois[j], rect) > params.trackIou) { dup = true; } }
    if (!dup) { rois[out] = rect; out++; }
  }
  for (var i = 0; i < out; i++) { lmCrops[i] = roiCrop(rois[i], 256.0); }
  lmBatch = vec4<i32>(out, 0, 0, 0);
}`;
}

function lmPostSource(o) {
  return `${COMMON}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lm: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> rois: array<vec4<f32>, ${MAX_FACES}>;
@group(0) @binding(3) var<storage, read> lmBatch: vec4<i32>;
@group(0) @binding(4) var<storage, read_write> track: Track;
@group(0) @binding(5) var<storage, read_write> ttBatch: vec4<i32>;
@group(0) @binding(6) var<storage, read_write> ttCrops: array<Crop, ${2 * MAX_FACES}>;
@group(0) @binding(7) var<storage, read_write> outb: array<f32, ${OUT_FLOATS}>;
const FS = ${FACE_STRIDE};
fn lmv(b: i32, c: i32) -> f32 { return bitcast<vec4<f32>>(lm[${o.pts.off} + b * ${o.pts.CG} + c / 4])[c % 4]; }
fn pt(f: i32, k: i32) -> vec2<f32> { return vec2<f32>(outb[4 + f * FS + 16 + 2 * k], outb[4 + f * FS + 17 + 2 * k]); }

// Upstream's crop: similarity transform putting eye and mouth midpoints on the template, mip chosen like drawCrop.
fn ttCrop(e: vec2<f32>, m: vec2<f32>, tE: vec2<f32>, tM: vec2<f32>) -> Crop {
  let d = m - e; let D = tM - tE; let n = dot(d, d);
  let re = (D.x * d.x + D.y * d.y) / n; let im = (D.y * d.x - D.x * d.y) / n;
  let tx = tE.x - (re * e.x - im * e.y); let ty = tE.y - (im * e.x + re * e.y);
  var s = length(vec2<f32>(re, im)); var lvl = 0;
  loop { if (s >= 0.25 || lvl >= min(6, params.levels - 1)) { break; } s *= 2.0; lvl++; }
  let sc = params.frame.x / levelSize(lvl).x; let r2 = re * sc; let i2 = im * sc;
  let E = tx + 0.5 - (r2 - i2) * 0.5; let F = ty + 0.5 - (i2 + r2) * 0.5; let det = r2 * r2 + i2 * i2;
  var o: Crop;
  o.m = vec4<f32>(r2 / det, -i2 / det, i2 / det, r2 / det);
  o.t = vec2<f32>(-(r2 * E + i2 * F) / det, (i2 * E - r2 * F) / det);
  o.lvl = lvl; o.pad = 0;
  return o;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let t = i32(li);
  let count = lmBatch.x;
  // Project every landmark from its 256x256 crop back into the frame.
  for (var f = 0; f < count; f++) {
    let r = rois[f]; let c = cos(r.w); let s = sin(r.w);
    for (var k = t; k < ${N_LM}; k += 256) {
      let u = vec2<f32>(lmv(f, 3 * k), lmv(f, 3 * k + 1)) / 256.0 - 0.5;
      let p = r.xy + vec2<f32>(c * u.x - s * u.y, s * u.x + c * u.y) * r.z;
      outb[4 + f * FS + 16 + 2 * k] = p.x; outb[4 + f * FS + 17 + 2 * k] = p.y;
    }
  }
  storageBarrier();
  workgroupBarrier();
  if (t != 0) { return; }
  var valid = 0;
  for (var f = 0; f < count; f++) {
    let presence = 1.0 / (1.0 + exp(-bitcast<vec4<f32>>(lm[${o.presence.off} + f * ${o.presence.CG}]).x));
    var lo = vec2<f32>(1e9); var hi = vec2<f32>(-1e9);
    for (var k = 0; k < ${N_LM}; k++) { let p = pt(f, k); lo = min(lo, p); hi = max(hi, p); }
    let l = (pt(f, 33) + pt(f, 133)) * 0.5; let rr = (pt(f, 362) + pt(f, 263)) * 0.5;
    let e = (l + rr) * 0.5; let m = (pt(f, 61) + pt(f, 291)) * 0.5;
    let eyes = length(rr - l); let span = length(m - e);
    let ok = presence >= params.minPresence && span >= 8.0 && eyes / span > 0.35 && eyes / span < 1.9;
    let base = 4 + f * FS;
    outb[base] = presence; outb[base + 1] = select(-1.0, f32(valid), ok);
    outb[base + 2] = e.x; outb[base + 3] = e.y; outb[base + 4] = m.x; outb[base + 5] = m.y;
    outb[base + 6] = rois[f].x; outb[base + 7] = rois[f].y; outb[base + 8] = rois[f].z; outb[base + 9] = rois[f].w;
    outb[base + 10] = lo.x; outb[base + 11] = lo.y; outb[base + 12] = hi.x; outb[base + 13] = hi.y;
    if (presence >= params.minPresence) {
      // Next frame's ROI from these landmarks: bounding box, rotation from landmark 33 to 263, square long x1.5.
      let a = pt(f, 33); let b = pt(f, 263);
      track.rects[valid] = vec4<f32>((lo + hi) * 0.5, max(hi.x - lo.x, hi.y - lo.y) * 1.5, normAngle(-atan2(-(b.y - a.y), b.x - a.x)));
    }
    if (ok) {
      ttCrops[2 * valid] = ttCrop(e, m, vec2<f32>(${EXP.E[0]}, ${EXP.E[1]}), vec2<f32>(${EXP.M[0]}, ${EXP.M[1]}));
      ttCrops[2 * valid + 1] = ttCrop(e, m, vec2<f32>(${AGE.E[0]}, ${AGE.E[1]}), vec2<f32>(${AGE.M[0]}, ${AGE.M[1]}));
    }
    if (presence >= params.minPresence) { valid++; }
  }
  // Tracked faces and scored faces use the same compacted order (a tracked face that fails the sanity filter keeps
  // its slot with an empty crop, so slot indices still line up).
  track.count = valid; track.pad0 = 0; track.pad1 = 0; track.pad2 = 0;
  ttBatch = vec4<i32>(valid, valid, 0, 0);
  outb[0] = f32(count); outb[1] = f32(valid);
}`;
}

export class FaceFinder {
  static async create(device, { base = 'models/', opt = {}, tuned = null, minDetection = 0.5, minPresence = 0.6 } = {}) {
    const ff = new FaceFinder();
    ff.device = device;
    const [detSpec, lmSpec] = await Promise.all([loadModel(base, 'face_detector'), loadModel(base, 'face_landmarks')]);
    const U = GPUBufferUsage;
    const mk = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    ff.lmBatch = mk(16, U.STORAGE);
    ff.lmCrops = mk(MAX_FACES * 32, U.STORAGE);
    ff.ttBatch = mk(16, U.STORAGE | U.COPY_DST);
    ff.ttCrops = mk(2 * MAX_FACES * 32, U.STORAGE | U.COPY_DST);
    ff.rois = mk(MAX_FACES * 16, U.STORAGE);
    ff.track = mk(16 + MAX_FACES * 16, U.STORAGE | U.COPY_DST);
    ff.out = mk(OUT_FLOATS * 4, U.STORAGE | U.COPY_SRC);
    ff.params = mk(64, U.UNIFORM | U.COPY_DST);
    [ff.det, ff.lm] = await Promise.all([
      createProgram(device, [detSpec], { maxBatch: 1, source: 'texture', opt, tuned, label: 'detector' }),
      createProgram(device, [lmSpec], { maxBatch: MAX_FACES, source: 'texture', opt, tuned, batchBuffer: ff.lmBatch, cropsBuffer: ff.lmCrops, label: 'landmarks' }),
    ]);
    const d = n => ff.det.output(0, n), l = n => ff.lm.output(0, n);
    const detPost = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: detPostSource({ cls16: d('t160'), cls8: d('t163'), reg16: d('t166'), reg8: d('t169') }), label: 'detPost' }), entryPoint: 'main' } });
    const lmPost = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: lmPostSource({ presence: l('t472'), pts: l('t473') }), label: 'lmPost' }), entryPoint: 'main' } });
    const bg = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ff.detPost = { p: detPost, bg: bg(detPost, [ff.params, ff.det.act, ff.track, ff.rois, ff.lmBatch, ff.lmCrops]) };
    ff.lmPost = { p: lmPost, bg: bg(lmPost, [ff.params, ff.lm.act, ff.rois, ff.lmBatch, ff.track, ff.ttBatch, ff.ttCrops, ff.out]) };
    ff.minDetection = minDetection; ff.minPresence = minPresence;
    ff.cropData = new ArrayBuffer(32);
    ff.frameKey = '';
    return ff;
  }

  // Forget tracked faces (e.g. after switching cameras).
  reset() { this.device.queue.writeBuffer(this.track, 0, new Int32Array(4)); }

  // Detector input: the whole frame letterboxed into 128x128 (uniform scale, centered), from a suitable mip.
  setFrame(w, h, levels) {
    const key = `${w}x${h}x${levels}`;
    if (key === this.frameKey) return;
    this.frameKey = key;
    const s = Math.max(w, h) / 128;
    let lvl = 0, sc = 1 / s;
    while (sc < 0.25 && lvl < levels - 1) { sc *= 2; lvl++; }
    let lw = w, lh = h;
    for (let i = 0; i < lvl; i++) { lw = Math.max(1, lw >> 1); lh = Math.max(1, lh >> 1); }
    const kx = lw / w, ky = lh / h;
    writeCrops(new Float32Array(this.cropData), new Int32Array(this.cropData), [[{ m: [s * kx, 0, 0, s * ky], t: [(w / 2 - 64 * s) * kx, (h / 2 - 64 * s) * ky], lvl }]], 1);
    this.device.queue.writeBuffer(this.det.crops, 0, this.cropData);
    const p = new ArrayBuffer(64), f = new Float32Array(p), i = new Int32Array(p);
    f.set([w, h, 128 * s, this.minDetection, w / 2 - 64 * s, h / 2 - 64 * s, this.minPresence, 0.3, 0.5]);
    i[9] = MAX_FACES; i[10] = levels;
    this.device.queue.writeBuffer(this.params, 0, p);
  }

  encode(enc, view) {
    this.det.encode(enc, { view, batch: [1] });
    let pass = enc.beginComputePass();
    pass.setPipeline(this.detPost.p); pass.setBindGroup(0, this.detPost.bg); pass.dispatchWorkgroups(1); pass.end();
    this.lm.encode(enc, { view });
    pass = enc.beginComputePass();
    pass.setPipeline(this.lmPost.p); pass.setBindGroup(0, this.lmPost.bg); pass.dispatchWorkgroups(1); pass.end();
  }

  // Decodes the output block (as read back) into faces in frame pixels.
  static decode(f) {
    const n = f[0], faces = [];
    for (let k = 0; k < n; k++) {
      const b = 4 + k * FACE_STRIDE;
      faces.push({ presence: f[b], slot: f[b + 1], e: [f[b + 2], f[b + 3]], m: [f[b + 4], f[b + 5]], roi: [f[b + 6], f[b + 7], f[b + 8], f[b + 9]],
        box: [f[b + 10], f[b + 11], f[b + 12], f[b + 13]], pts: f.subarray(b + 16, b + 16 + 2 * N_LM) });
    }
    return faces;
  }
}

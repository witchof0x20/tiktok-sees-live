// Bit-exactness check: the reference CPU engine and the fused WebGPU engine must agree on every output.
import { requestDevice, createEngine } from '../gpu/engine.js';
import { makePhoto, drawCrop, EXP, AGE, findFaces } from '/upstream/tools/tiktok-face/face.js';

const BASE = '/upstream/tools/tiktok-face/';
const log = (t, cls) => { const el = document.getElementById('log'); if (el.textContent === 'running…') el.textContent = ''; const s = document.createElement('span'); if (cls) s.className = cls; s.textContent = t + '\n'; el.append(s); console.log(t); };
const WANT = { expnet: ['predict_attractive', 'prob_exp', 'predict_valence', 'predict_arousal', 'predict_happy'], agenet: ['predict_age', 'prob_gender'] };

async function load(name) {
  const [spec, buf] = await Promise.all([fetch(BASE + name + '.json').then(r => r.json()), fetch(BASE + name + '.bin').then(r => r.arrayBuffer())]);
  return { spec, buf, wanted: WANT[name] };
}
function tensorOf(t, p, e, m) {
  const c = document.createElement('canvas'); c.width = c.height = t.S;
  const x = c.getContext('2d', { willReadFrequently: true }); drawCrop(x, t.S, t, p, e, m);
  const d = x.getImageData(0, 0, t.S, t.S).data, o = new Int16Array(t.S * t.S * 3);
  for (let i = 0; i < t.S * t.S; i++) { o[i * 3] = d[i * 4 + 2] - 128; o[i * 3 + 1] = d[i * 4 + 1] - 128; o[i * 3 + 2] = d[i * 4] - 128; }
  return o;
}
function synth(S, kind, seed) {
  let s = seed >>> 0; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const o = new Int16Array(S * S * 3);
  for (let i = 0; i < o.length; i++) {
    const p = (i / 3) | 0, y = (p / S) | 0, x = p % S;
    o[i] = kind === 'noise' ? Math.floor(rnd() * 256) - 128 : kind === 'max' ? 127 : kind === 'min' ? -128 : kind === 'checker' ? (((x >> 2) ^ (y >> 2)) & 1 ? 127 : -128) : Math.round(127 * Math.sin(x * 0.07 + seed) * Math.cos(y * 0.05 + i % 3));
  }
  return o;
}
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

(async () => {
  try {
    const { device, info } = await requestDevice();
    log(`adapter: ${info.vendor} ${info.architecture} ${info.description || ''}`);
    const specs = await Promise.all([load('expnet'), load('agenet')]);
    const ref = specs.map(s => new self.FaceNet.Net(s.spec, s.buf));
    const opt = JSON.parse(new URLSearchParams(location.search).get('opt') || '{}');
    const eng = await createEngine(device, specs, { maxBatch: 8, source: 'tensor', opt });
    log(`compiled ${eng.kernelCount} kernels into ${eng.dispatches} dispatches in ${eng.compileMs.toFixed(0)} ms; arena ${(eng.arenaBytes / 1024).toFixed(0)} KiB, weights ${(eng.weightBytes / 1024).toFixed(0)} KiB`);

    const cases = [];
    const img = new Image(); img.src = BASE + 'sample.jpg'; await img.decode();
    const photo = makePhoto(img), faces = await findFaces(photo);
    log(`sample.jpg faces: ${faces.length}`);
    for (const f of faces) cases.push({ name: 'sample face', t: [tensorOf(EXP, photo, f.e, f.m), tensorOf(AGE, photo, f.e, f.m)] });
    for (const [k, seed] of [['noise', 1], ['noise', 2], ['max', 0], ['min', 0], ['checker', 0], ['wave', 3]]) cases.push({ name: `${k}#${seed}`, t: [synth(256, k, seed), synth(224, k, seed + 7)] });

    let bad = 0;
    const got = (await eng.runTensors(cases.map(c => c.t))).res;
    cases.forEach((c, b) => {
      const R = [ref[0].run(c.t[0]), ref[1].run(c.t[1])];
      const diffs = [];
      specs.forEach((s, i) => s.wanted.forEach(o => { if (!same(Array.from(R[i][o].v), Array.from(got[b][i][o].v))) diffs.push(`${o}: ref ${Array.from(R[i][o].v).map(v => v.toFixed(6))} gpu ${Array.from(got[b][i][o].v).map(v => v.toFixed(6))}`); }));
      if (diffs.length) { bad++; log(`MISMATCH ${c.name}\n  ${diffs.join('\n  ')}`, 'bad'); }
      else log(`exact ${c.name}: attractive ${R[0].predict_attractive.v[0].toFixed(4)} age ${(R[1].predict_age.v[0] * 10).toFixed(2)}`, 'ok');
    });
    log(bad ? `${bad} case(s) differ` : 'ALL OUTPUTS BIT-IDENTICAL', bad ? 'bad' : 'ok');
    window.__verify = { bad, cases: cases.length };
  } catch (err) { log(String(err.stack || err), 'bad'); window.__verify = { error: String(err) }; }
})();

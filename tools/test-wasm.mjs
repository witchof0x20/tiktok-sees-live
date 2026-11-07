// Node check of the Rust/WASM engine: bit-exact against the original reference engine (upstream/, from
// tools/fetch-upstream.sh) on noise and extreme inputs, plus timing.
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { createWasmNet } from '../live/cpu/wasm-net.js';
const require = createRequire(import.meta.url);
const root = new URL('..', import.meta.url).pathname;
const ref = require(root + 'upstream/tools/tiktok-face/engine.js');
const mod = await WebAssembly.compile(fs.readFileSync(root + (process.argv[2] || 'wasm/target/wasm32-unknown-unknown/release/tiktok_cpu.wasm')));
const WANT = { expnet: ['predict_attractive', 'prob_exp', 'predict_valence', 'predict_arousal', 'predict_happy'], agenet: ['predict_age', 'prob_gender'] };
const load = n => { const b = fs.readFileSync(root + `live/nets/${n}.bin`); return { spec: JSON.parse(fs.readFileSync(root + `live/nets/${n}.json`)), buf: b.buffer.slice(b.byteOffset, b.byteOffset + b.length) }; };
let bad = 0;
for (const name of ['expnet', 'agenet']) {
  const { spec, buf } = load(name), S = spec.ops[0].h;
  const wnet = await createWasmNet(mod, spec, buf, WANT[name]), rnet = new ref.Net(spec, buf);
  const cases = { noise: null, max: 127, min: -128, checker: 'c' };
  let s = 7;
  for (const [k, v] of Object.entries(cases)) {
    const t = new Int16Array(S * S * 3);
    for (let i = 0; i < t.length; i++) { const p = (i / 3) | 0; s = (s * 1664525 + 1013904223) >>> 0; t[i] = v === null ? (s >>> 24) - 128 : v === 'c' ? ((((p % S) >> 3) ^ ((p / S | 0) >> 3)) & 1 ? 127 : -128) : v; }
    const R = rnet.run(t), G = wnet.run(t);
    const diff = WANT[name].filter(o => Array.from(R[o].v).some((x, i) => !Object.is(x, G[o].v[i])));
    if (diff.length) { bad++; console.log(`MISMATCH ${name} ${k}: ${diff}`); } else console.log(`exact ${name} ${k}`);
  }
  const t = new Int16Array(S * S * 3).map((_, i) => (i * 37) % 256 - 128);
  for (let i = 0; i < 3; i++) wnet.run(t);
  const t0 = performance.now(); for (let i = 0; i < 20; i++) wnet.run(t); const ms = (performance.now() - t0) / 20;
  const t1 = performance.now(); rnet.run(t); const rms = performance.now() - t1;
  console.log(`  ${name}: wasm ${ms.toFixed(1)} ms per face, reference JS ${rms.toFixed(0)} ms`);
}
console.log(bad ? `${bad} mismatches` : 'ALL BIT-IDENTICAL');
process.exit(bad ? 1 : 0);

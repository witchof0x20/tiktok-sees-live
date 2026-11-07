// Module worker running one TikTok net on the CPU (Rust/WebAssembly engine). Two of these run the nets in parallel.
import { createWasmNet } from './wasm-net.js';

let net = null;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const [module, spec, buf] = await Promise.all([
        WebAssembly.compileStreaming(fetch(data.wasm)),
        fetch(data.spec).then(r => r.json()), fetch(data.bin).then(r => r.arrayBuffer())]);
      net = await createWasmNet(module, spec, buf, data.wanted);
      postMessage({ type: 'ready' });
    } else if (data.type === 'run') {
      const t0 = performance.now();
      // One input per face; outputs as plain { name: { v: [...] } }.
      const out = data.inputs.map(t => Object.fromEntries(Object.entries(net.run(t)).map(([k, v]) => [k, { v: Array.from(v.v) }])));
      postMessage({ type: 'result', id: data.id, out, ms: performance.now() - t0 });
    }
  } catch (err) { postMessage({ type: 'error', id: data.id, message: String(err?.message || err) }); }
};

// The TikTok nets on the CPU via the Rust/WebAssembly engine (wasm/src/lib.rs). Builds the engine's program format
// from the parsed graph (same parser as the GPU path); the Rust side validates it fully before running. Pooled sums
// go through the same exact float tail (runTail) as the GPU path, so outputs are bit-identical to the reference.
import { parseNet, runTail } from '../gpu/graph.js';

const MAGIC = 0x544b4e31;
const rsBias = (x, s) => s <= 0 ? x * 2 ** -s : Math.floor(x / 2 ** s + 0.5);

export function buildProgram(spec, buf, wanted) {
  const net = parseNet(spec, wanted);
  const words = [], w = [], b = [];
  const ch = n => n.C + (n.C & 1); // the input tensor's 3 channels are padded to 4
  let nSums = 0;
  for (const g of net.gaps) { g.off = nSums; nSums += net.nodes.get(g.src).C; }
  const ops = [];
  for (const n of net.order) {
    if (n.op === 'data') continue;
    const X = n.ins[0];
    if (n.op === 'conv' || n.op === 'dw') {
      const o = n.spec, taps = o.kh * o.kw, src = new Int16Array(buf, o.wOff, n.op === 'dw' ? taps * o.o : o.o * taps * o.i);
      const bt = o.bias ? new Int32Array(buf, o.bOff, o.o) : null, bs = o.bf - (X.f + o.wf), bOff = b.length, wOff = w.length;
      for (let c = 0; c < o.o; c++) b.push(bt ? rsBias(bt[c], bs) : 0);
      const icp = ch(X);
      if (n.op === 'dw') {
        for (let c = 0; c < o.o; c++) for (let t = 0; t < taps; t++) w.push(src[t * o.o + c]);
        ops.push([3, n.idx, X.idx, o.kh, o.kw, n.s, n.p, n.sh, n.act ? 1 : 0, wOff, bOff]);
      } else if (o.kh === 1 && o.kw === 1 && n.p === 0) {
        for (let c = 0; c < o.o; c++) for (let i = 0; i < icp; i++) w.push(i < o.i ? src[c * o.i + i] : 0);
        ops.push([1, n.idx, X.idx, n.s, n.sh, n.act ? 1 : 0, wOff, bOff]);
      } else {
        for (let c = 0; c < o.o; c++) for (let t = 0; t < taps; t++) for (let i = 0; i < icp; i++) w.push(i < o.i ? src[(c * taps + t) * o.i + i] : 0);
        ops.push([2, n.idx, X.idx, o.kh, o.kw, n.s, n.p, n.sh, n.act ? 1 : 0, wOff, bOff]);
      }
    } else if (n.op === 'add') ops.push([4, n.idx, n.ins[0].idx, n.ins[1].idx, n.sa, n.sb, n.act ? 1 : 0]);
    else if (n.op === 'concat') ops.push([5, n.idx, n.ins.length, ...n.ins.flatMap((x, i) => [x.idx, n.shifts[i]])]);
    if (n.toGap) ops.push([6, n.idx, net.gaps.find(g => g.src === n.name).off]);
  }
  words.push(MAGIC, 1, net.order.length, ops.length, net.data.idx, nSums);
  for (const n of net.order) words.push(ch(n), n.H, n.W);
  for (const op of ops) words.push(...op);
  if (words.some(v => !Number.isInteger(v) || v < 0 || v > 0xffffffff)) throw new Error('program value out of range');
  return { net, words: Uint32Array.from(words), w: Int16Array.from(w), b: Int32Array.from(b), nSums };
}

// module: a compiled WebAssembly.Module of tiktok_cpu.wasm (compile once, instantiate per net).
export async function createWasmNet(module, spec, buf, wanted) {
  const { instance } = await WebAssembly.instantiate(module, {}).then(i => ({ instance: i.instance ?? i }));
  const x = instance.exports, prog = buildProgram(spec, buf, wanted);
  const put = arr => {
    const p = x.alloc_bytes(arr.byteLength);
    new Uint8Array(x.memory.buffer, p, arr.byteLength).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    return p;
  };
  const p = x.prog_load(put(prog.words), prog.words.length, put(prog.w), prog.w.length, put(prog.b), prog.b.length);
  if (!p) throw new Error(`the wasm engine rejected the ${prog.net.name} program`);
  const d = prog.net.data, n = d.H * d.W * d.C;
  return {
    net: prog.net,
    // input: HWC Int16Array (BGR - 128), like the reference tensor. Returns the tail outputs ({ name: { v } }).
    run(input) {
      if (input.length !== n) throw new Error('bad input size');
      new Int16Array(x.memory.buffer, x.prog_input(p), n).set(input);
      x.prog_run(p);
      const sums = new Int32Array(x.memory.buffer, x.prog_sums(p), prog.nSums).slice();
      return runTail(prog.net, buf, sums, 0);
    },
  };
}

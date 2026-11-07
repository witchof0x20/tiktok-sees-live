// Turns a decoded net spec into a typed graph: shapes, fractional bits, consumers, and dead-code elimination.
// Everything after the global average pools (flat/fc/softmax) stays on the CPU as the "tail".

export const SPATIAL = new Set(['conv', 'dw', 'pool']);
export const POINT = new Set(['add', 'concat', 'cpad', 'act']);

export function parseNet(spec, wanted) {
  const ops = spec.ops, byOut = new Map();
  for (const o of ops) byOut.set(o.bot ?? o.name, o);
  const inputsOf = o => o.op === 'data' ? [] : o.op === 'add' ? [o.a, o.b] : o.op === 'concat' ? o.tops : [o.top];

  // Live set: walk back from the outputs the caller actually reads.
  const live = new Set(), stack = [...wanted];
  while (stack.length) {
    const n = stack.pop();
    if (live.has(n)) continue;
    if (!byOut.has(n)) throw new Error(`unknown output ${n}`);
    live.add(n);
    stack.push(...inputsOf(byOut.get(n)));
  }

  const nodes = new Map(), order = [], tail = [], gaps = [];
  for (const o of ops) {
    const name = o.bot ?? o.name;
    if (!live.has(name)) continue;
    if (o.op === 'gap' || o.op === 'flat' || o.op === 'fc' || o.op === 'softmax') {
      tail.push(o);
      if (o.op === 'gap') gaps.push({ name, src: o.top });
      continue;
    }
    const ins = inputsOf(o).map(i => nodes.get(i));
    if (ins.some(x => !x)) throw new Error(`input of ${name} not on the GPU side`);
    const n = { name, op: o.op, ins, cons: [], spec: o, idx: order.length };
    if (o.op === 'data') { n.C = o.c; n.H = o.h; n.W = o.w; n.f = o.frac; }
    else if (SPATIAL.has(o.op)) {
      const X = ins[0];
      if (o.op === 'dw' && o.o !== X.C) throw new Error(`dw channel mismatch at ${name}`);
      if (o.op === 'conv' && o.i !== X.C) throw new Error(`conv channel mismatch at ${name}`);
      n.C = o.o; n.H = Math.floor((X.H + 2 * o.p - o.kh) / o.s) + 1; n.W = Math.floor((X.W + 2 * o.p - o.kw) / o.s) + 1; n.f = o.of;
      n.k = [o.kh, o.kw]; n.s = o.s; n.p = o.p; n.act = !!o.act; n.sh = X.f + o.wf - o.of;
    } else if (o.op === 'add') {
      const [A, B] = ins;
      if (A.C !== B.C || A.H !== B.H || A.W !== B.W) throw new Error(`add shape mismatch at ${name}`);
      n.C = A.C; n.H = A.H; n.W = A.W; n.f = o.of; n.act = !!o.act; n.sa = A.f - o.of; n.sb = B.f - o.of;
    } else if (o.op === 'concat') {
      const F = ins[0];
      if (ins.some(x => x.H !== F.H || x.W !== F.W)) throw new Error(`concat shape mismatch at ${name}`);
      n.C = ins.reduce((a, x) => a + x.C, 0); n.H = F.H; n.W = F.W; n.f = o.of; n.shifts = ins.map(x => x.f - o.of);
      if (ins.some(x => x.C % 4)) throw new Error(`concat part not a multiple of 4 at ${name}`);
    } else throw new Error(`unsupported op ${o.op}`);
    if (o.op !== 'data' && n.C % 4) throw new Error(`channel count ${n.C} not a multiple of 4 at ${name}`);
    n.CG = Math.ceil(n.C / 4);
    for (const x of ins) x.cons.push(n);
    nodes.set(name, n); order.push(n);
  }
  const gapSrc = new Set(gaps.map(g => g.src));
  for (const n of order) n.toGap = gapSrc.has(n.name);
  for (const n of order) if (n.toGap && n.cons.length) throw new Error(`${n.name} feeds a pool and another op`);
  return { name: spec.name, nodes, order, tail, gaps, data: order[0] };
}

// Packs weights into the NC4 layouts the kernels read: conv [tap][icPad][oc/4] vec4, dw [tap][c/4] vec4, bias [c/4] vec4.
// Bias is pre-shifted onto the accumulator's scale exactly the way the reference engine does it.
// With u24, 1x1/kxk conv weights are stored offset (w + 2048) and the bias absorbs the offset terms, so the
// kernels can multiply unsigned 16-bit x 12-bit operands (full-rate 24-bit multiply) and still be exact mod 2^32.
export function packWeights(net, buf, { u24 = true } = {}) {
  const rs = (x, s) => s <= 0 ? x * Math.pow(2, -s) : Math.floor(x / Math.pow(2, s) + 0.5);
  const chunks = [];
  let off = 0; // in vec4 units
  for (const n of net.order) {
    if (n.op !== 'conv' && n.op !== 'dw') continue;
    const o = n.spec, X = n.ins[0], taps = o.kh * o.kw, dw = n.op === 'dw';
    let w;
    if (dw) {
      const src = new Int16Array(buf, o.wOff, taps * o.o);
      w = new Int32Array(taps * n.CG * 4);
      for (let t = 0; t < taps; t++) for (let c = 0; c < o.o; c++) w[t * n.CG * 4 + c] = src[t * o.o + c];
    } else {
      const src = new Int16Array(buf, o.wOff, o.o * taps * o.i), icPad = X.CG * 4;
      n.u24 = u24 && src.every(v => v >= -2048 && v <= 2047);
      w = new Int32Array(taps * icPad * n.CG * 4);
      if (n.u24) w.fill(2048);
      for (let c = 0; c < o.o; c++) for (let t = 0; t < taps; t++) for (let i = 0; i < o.i; i++)
        w[(t * icPad + i) * n.CG * 4 + c] = src[(c * taps + t) * o.i + i] + (n.u24 ? 2048 : 0);
    }
    const b = new Int32Array(n.CG * 4);
    if (o.bias) {
      const bt = new Int32Array(buf, o.bOff, o.o), s = o.bf - (X.f + o.wf);
      for (let c = 0; c < o.o; c++) {
        const v = rs(bt[c], s);
        if (!Number.isSafeInteger(v) || Math.abs(v) >= 2 ** 31) throw new Error(`bias out of range at ${n.name}`);
        b[c] = v;
      }
    }
    if (n.u24) {
      // sum (x'-32768)(w'-2048) = sum x'w' - 2048 sum x' - 32768 sum w' + K 2^26; the kernel subtracts 2048 sum x'.
      const K = taps * X.CG * 4, cog = n.CG * 4;
      for (let c = 0; c < o.o; c++) {
        let sw = 0;
        for (let t = 0; t < taps * X.CG * 4; t++) sw += w[t * cog + c];
        const v = b[c] - 32768 * sw + K * 2 ** 26;
        b[c] = ((v % 2 ** 32) + 2 ** 32) % 2 ** 32 | 0;
      }
    }
    n.bOff = off; chunks.push(b); off += b.length / 4;
    n.wOff = off; chunks.push(w); off += w.length / 4;
  }
  const all = new Int32Array(off * 4);
  let p = 0;
  for (const c of chunks) { all.set(c, p); p += c.length; }
  return all;
}

// The CPU tail, written exactly like the reference engine so the float outputs are bit-identical.
export function runTail(net, buf, sums, b) {
  {
    const T = {};
    for (const g of net.gaps) {
      const src = net.nodes.get(g.src), sc = Math.pow(2, src.f), n = src.H * src.W, v = new Float64Array(src.C);
      for (let c = 0; c < src.C; c++) v[c] = sums[g.off + b * src.C + c] / n / sc;
      T[g.name] = { v };
    }
    for (const o of net.tail) {
      if (o.op === 'gap') continue;
      if (o.op === 'flat') T[o.bot] = T[o.top];
      else if (o.op === 'fc') {
        const x = T[o.top].v, w = new Float32Array(buf, o.wOff, o.o * o.i), bb = new Float32Array(buf, o.bOff, o.o), v = new Float64Array(o.o);
        for (let j = 0; j < o.o; j++) { let s = bb[j]; for (let i = 0; i < o.i; i++) s += w[j * o.i + i] * x[i]; v[j] = s; }
        T[o.bot] = { v };
      } else if (o.op === 'softmax') {
        const x = T[o.top].v; let m = -Infinity; for (const a of x) m = Math.max(m, a);
        let s = 0; const v = x.map(a => { const e = Math.exp(a - m); s += e; return e; });
        T[o.bot] = { v: v.map(e => e / s) };
      }
    }
    return T;
  }
}

// Float graphs converted from TFLite (tools/convert_tflite.py). Channel counts need not be multiples of 4: tensors
// are padded to whole channel groups and the padding channels stay exactly zero (zero weights and biases).
export const FLOAT_POINT = new Set(['add', 'cpad', 'act']);
export function parseFloatNet(ir, wanted) {
  const byName = new Map(ir.nodes.map(n => [n.name, n]));
  const live = new Set(), stack = [...wanted];
  while (stack.length) {
    const n = stack.pop();
    if (live.has(n)) continue;
    if (!byName.has(n)) throw new Error(`unknown output ${n}`);
    live.add(n);
    stack.push(...(byName.get(n).ins || []));
  }
  const nodes = new Map(), order = [];
  for (const o of ir.nodes) {
    if (!live.has(o.name)) continue;
    const ins = (o.ins || []).map(i => nodes.get(i));
    if (ins.some(x => !x)) throw new Error(`input of ${o.name} missing`);
    const n = { name: o.name, op: o.op, ins, cons: [], spec: o, idx: order.length, dtype: 'f32', C: o.C, H: o.H, W: o.W, CG: Math.ceil(o.C / 4), act: o.act || null };
    if (o.op === 'data') n.range = o.range;
    else if (o.op === 'conv' || o.op === 'dw' || o.op === 'pool') {
      Object.assign(n, { k: o.k, s: o.s, p: o.p });
      const X = ins[0];
      if (o.op !== 'conv' && o.C !== X.C) throw new Error(`${o.op} channel mismatch at ${o.name}`);
      const span = (d, out) => (out - 1) * o.s - o.p + d;
      if (span(o.k[0], o.H) > X.H + o.p + 1 || o.H < 1) throw new Error(`bad geometry at ${o.name}`);
    } else if (o.op === 'add') {
      if (ins[0].C !== ins[1].C || ins[0].H !== ins[1].H) throw new Error(`add shape mismatch at ${o.name}`);
    } else if (o.op === 'cpad') {
      if (o.C < ins[0].C || ins[0].H !== o.H) throw new Error(`bad cpad at ${o.name}`);
    } else if (o.op !== 'act') throw new Error(`unsupported op ${o.op}`);
    if (n.act && !['relu', 'prelu'].includes(n.act)) throw new Error(`unsupported activation ${n.act}`);
    for (const x of ins) x.cons.push(n);
    nodes.set(o.name, n); order.push(n);
  }
  for (const n of order) n.toGap = false;
  const outputs = wanted.map(w => nodes.get(w));
  return { name: ir.name, dtype: 'f32', nodes, order, tail: [], gaps: [], data: order[0], outputs };
}

// Packs float weights (TFLite OHWI / 1HWC) into the NC4 layouts the kernels read, as f32 bit patterns.
export function packFloatWeights(net, buf) {
  const src = new Float32Array(buf), chunks = [];
  let off = 0;
  const push = arr => { const at = off; chunks.push(arr); off += arr.length / 4; return at; };
  const vecs = (base, C, CG) => { const a = new Float32Array(CG * 4); for (let c = 0; c < C; c++) a[c] = src[base + c]; return a; };
  for (const n of net.order) {
    const o = n.spec;
    if (n.op === 'conv' || n.op === 'dw') {
      const X = n.ins[0], [kh, kw] = n.k, taps = kh * kw, cog = n.CG;
      let w;
      if (n.op === 'dw') {
        w = new Float32Array(taps * cog * 4);
        for (let t = 0; t < taps; t++) for (let c = 0; c < n.C; c++) w[t * cog * 4 + c] = src[o.wOff + t * n.C + c];
      } else {
        const icPad = X.CG * 4;
        w = new Float32Array(taps * icPad * cog * 4);
        for (let c = 0; c < n.C; c++) for (let t = 0; t < taps; t++) for (let i = 0; i < X.C; i++)
          w[(t * icPad + i) * cog * 4 + c] = src[o.wOff + (c * taps + t) * X.C + i];
      }
      n.bOff = push(vecs(o.bOff, n.C, cog));
      n.wOff = push(w);
    }
    if (n.act === 'prelu') n.alphaOff = push(vecs(o.aOff, n.C, n.CG));
  }
  const all = new Float32Array(off * 4);
  let p = 0;
  for (const c of chunks) { all.set(c, p); p += c.length; }
  return new Int32Array(all.buffer);
}

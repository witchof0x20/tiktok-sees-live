// Turns a decoded net spec into a typed graph: shapes, fractional bits, consumers, and dead-code elimination.
// Everything after the global average pools (flat/fc/softmax) stays on the CPU as the "tail".

export const SPATIAL = new Set(['conv', 'dw']);
export const POINT = new Set(['add', 'concat']);

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
export function packWeights(net, buf) {
  const rs = (x, s) => s <= 0 ? x * Math.pow(2, -s) : Math.floor(x / Math.pow(2, s) + 0.5);
  const chunks = [];
  let off = 0; // in vec4 units
  for (const n of net.order) {
    if (!SPATIAL.has(n.op)) continue;
    const o = n.spec, X = n.ins[0], taps = o.kh * o.kw, dw = n.op === 'dw';
    let w;
    if (dw) {
      const src = new Int16Array(buf, o.wOff, taps * o.o);
      w = new Int32Array(taps * n.CG * 4);
      for (let t = 0; t < taps; t++) for (let c = 0; c < o.o; c++) w[t * n.CG * 4 + c] = src[t * o.o + c];
    } else {
      const src = new Int16Array(buf, o.wOff, o.o * taps * o.i), icPad = X.CG * 4;
      w = new Int32Array(taps * icPad * n.CG * 4);
      for (let c = 0; c < o.o; c++) for (let t = 0; t < taps; t++) for (let i = 0; i < o.i; i++)
        w[(t * icPad + i) * n.CG * 4 + c] = src[(c * taps + t) * o.i + i];
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

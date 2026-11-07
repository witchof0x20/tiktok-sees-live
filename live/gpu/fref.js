// Plain CPU interpreter for the converted float graphs (NHWC, float64 accumulation). Slow; used only to check the
// GPU engine.
export function runFloatRef(ir, buf, input) {
  const W = new Float32Array(buf), T = new Map();
  const act = (n, v, c) => n.act === 'relu' ? Math.max(v, 0) : n.act === 'prelu' ? (v >= 0 ? v : v * W[n.aOff + c]) : v;
  for (const n of ir.nodes) {
    const { H, W: Wd, C } = n, out = new Float64Array(H * Wd * C);
    if (n.op === 'data') { T.set(n.name, Float64Array.from(input)); continue; }
    const X = n.ins.map(i => ({ t: T.get(i), node: ir.nodes.find(m => m.name === i) }));
    const x0 = X[0], xi = x0.node;
    if (n.op === 'conv' || n.op === 'dw' || n.op === 'pool') {
      const [kh, kw] = n.k;
      for (let y = 0; y < H; y++) for (let x = 0; x < Wd; x++) for (let c = 0; c < C; c++) {
        let s = n.op === 'pool' ? -Infinity : W[n.bOff + c];
        for (let ky = 0; ky < kh; ky++) for (let kx = 0; kx < kw; kx++) {
          const iy = y * n.s - n.p + ky, ix = x * n.s - n.p + kx;
          if (iy < 0 || iy >= xi.H || ix < 0 || ix >= xi.W) continue;
          const base = (iy * xi.W + ix) * xi.C;
          if (n.op === 'pool') s = Math.max(s, x0.t[base + c]);
          else if (n.op === 'dw') s += x0.t[base + c] * W[n.wOff + (ky * kw + kx) * C + c];
          else for (let i = 0; i < xi.C; i++) s += x0.t[base + i] * W[n.wOff + ((c * kh + ky) * kw + kx) * xi.C + i];
        }
        out[(y * Wd + x) * C + c] = Math.fround(act(n, s, c));
      }
    } else if (n.op === 'add') {
      const [a, b] = X;
      for (let p = 0; p < H * Wd; p++) for (let c = 0; c < C; c++) out[p * C + c] = Math.fround(act(n, a.t[p * C + c] + b.t[p * C + c], c));
    } else if (n.op === 'cpad') {
      for (let p = 0; p < H * Wd; p++) for (let c = 0; c < xi.C; c++) out[p * C + c] = x0.t[p * xi.C + c];
    } else if (n.op === 'act') {
      for (let p = 0; p < H * Wd; p++) for (let c = 0; c < C; c++) out[p * C + c] = Math.fround(act(n, x0.t[p * C + c], c));
    } else throw new Error(n.op);
    T.set(n.name, out);
  }
  return T;
}

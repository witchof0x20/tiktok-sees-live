// Fusion planner. Groups ops into kernels, decides which intermediates live in workgroup memory (with halos)
// versus being recomputed inline, picks tile shapes with a small cost model, then levelizes kernels so
// independent ones (other branches, the other net) can share one dispatch.
import { SPATIAL, POINT } from './graph.js';

export const DEFAULTS = {
  wg: 128,            // threads per workgroup, shared by every kernel so they can be fused horizontally
  sharedBytes: 16384, // workgroup memory budget per kernel
  concThreads: 8192,  // rough count of threads the GPU keeps in flight
  dispatchCost: 4000, // fixed cost per kernel, in the same units as per-thread work
  barrierCost: 40,
  maxRecompute: 2.5,  // reject fusions that redo more than this much of the unfused work
  tiles: [1, 2, 4, 7, 8, 14, 16, 28, 32],
  splits: [1, 2, 4, 8, 16],
  blocks: [[1, 1], [2, 1], [4, 1], [1, 2], [2, 2], [4, 2], [1, 4], [2, 4]],
  load: 4,            // relative cost of a vec4 load
  mad: 4,             // relative cost of a vec4 integer multiply-add
};

const ceil = (a, b) => Math.ceil(a / b);

// Which members get materialized in workgroup memory. Pointwise ops are always recomputed where read;
// a conv is recomputed only when every path from it to the kernel output is pointwise (so it's read once,
// at the same pixel). Anything a spatial op reads through needs halos, so it's staged in shared memory.
function modes(members, F) {
  const memo = new Map();
  const pwOnly = n => {
    if (n === F) return true;
    if (memo.has(n)) return memo.get(n);
    const r = POINT.has(n.op) && n.cons.every(c => members.has(c) && pwOnly(c));
    memo.set(n, r);
    return r;
  };
  const mode = new Map();
  for (const n of members) {
    if (n === F) mode.set(n, 'final');
    else if (POINT.has(n.op)) mode.set(n, 'inline');
    else mode.set(n, n.cons.every(c => POINT.has(c.op) && pwOnly(c)) ? 'inline' : 'shared');
  }
  return mode;
}

// Region each node is needed over, relative to its own tile origin (= kernel tile origin * resolution ratio).
function regions(order, F, T) {
  const R = new Map([[F, { y0: 0, y1: T, x0: 0, x1: T, scale: 1 }]]);
  for (let i = order.length - 1; i >= 0; i--) {
    const n = order[i], r = R.get(n);
    if (!r) continue;
    for (const x of n.ins) {
      const scale = SPATIAL.has(n.op) ? r.scale * n.s : r.scale;
      let q;
      if (SPATIAL.has(n.op)) q = { y0: r.y0 * n.s - n.p, y1: (r.y1 - 1) * n.s - n.p + n.k[0], x0: r.x0 * n.s - n.p, x1: (r.x1 - 1) * n.s - n.p + n.k[1] };
      else q = { ...r };
      q.scale = scale;
      const o = R.get(x);
      if (o && o.scale !== scale) return null;
      R.set(x, o ? { y0: Math.min(o.y0, q.y0), y1: Math.max(o.y1, q.y1), x0: Math.min(o.x0, q.x0), x1: Math.max(o.x1, q.x1), scale } : q);
    }
  }
  return R;
}

// Per-thread cost (rough instruction count) of producing one block of Q channel groups x P pixels.
export function blockCost(members, mode, opt) {
  const { load: LD, mad: MAD } = opt;
  const memoE = new Map(), memoB = new Map();
  const isRead = n => !members.has(n) || mode.get(n) === 'shared';
  const elem = n => {
    if (isRead(n)) return LD;
    if (memoE.has(n)) return memoE.get(n);
    let c;
    if (n.op === 'add') c = elem(n.ins[0]) + elem(n.ins[1]) + 3;
    else if (n.op === 'concat') c = Math.max(...n.ins.map(elem)) + 2;
    else c = blk(n, 1, 1);
    memoE.set(n, c);
    return c;
  };
  const blk = (n, P, Q) => {
    const key = `${n.idx}_${P}_${Q}`;
    if (memoB.has(key)) return memoB.get(key);
    let c;
    if (isRead(n)) c = P * Q * LD;
    else if (n.op === 'add') c = blk(n.ins[0], P, Q) + blk(n.ins[1], P, Q) + 3 * P * Q;
    else if (n.op === 'concat') c = Math.max(...n.ins.map(x => blk(x, P, Q))) + 2 * P * Q;
    else if (n.op === 'conv') c = n.k[0] * n.k[1] * n.ins[0].CG * (P * elem(n.ins[0]) + 4 * Q * LD + 4 * P * Q * MAD) + 4 * P * Q;
    else if (n.op === 'dw') c = n.k[0] * Q * (((P - 1) * n.s + n.k[1]) * elem(n.ins[0]) + n.k[1] * (LD + P * MAD)) + 4 * P * Q;
    memoB.set(key, c);
    return c;
  };
  return blk;
}

export const usefulMacs = n => n.op === 'conv' ? n.C * n.H * n.W * n.k[0] * n.k[1] * n.ins[0].C : n.op === 'dw' ? n.C * n.H * n.W * n.k[0] * n.k[1] : 0;

// Baseline: every op its own kernel, every tensor in global memory.
function soloWork(n, opt) {
  const m = new Set([n]);
  return n.CG * n.H * n.W * blockCost(m, modes(m, n), opt)(n, 1, 1);
}

const divides = (q, n) => n % q === 0;

export function planGroup(members, F, opt) {
  const order = [...members].sort((a, b) => a.idx - b.idx);
  const mode = modes(members, F);
  const cost = blockCost(members, mode, opt);
  const shared = order.filter(n => mode.get(n) === 'shared');
  // Channel blocking along the output path must not straddle a concat part.
  const path = order.filter(n => mode.get(n) !== 'shared');
  const qOk = q => path.every(n => divides(q, n.CG) && (n.op !== 'concat' || n.ins.every(x => divides(q, x.CG))));
  const base = order.reduce((a, n) => a + soloWork(n, opt), 0);
  let best = null;
  for (const T of new Set([...opt.tiles.filter(t => t < F.H), F.H])) {
    const R = regions(order, F, T);
    if (!R) return null;
    let bytes = F.toGap ? F.C * 4 : 0;
    for (const n of shared) { const r = R.get(n); bytes += n.CG * (r.y1 - r.y0) * (r.x1 - r.x0) * 16; }
    if (bytes > opt.sharedBytes) continue;
    const tiles = ceil(F.H, T) * ceil(F.W, T);
    for (const S of opt.splits) {
      if (!divides(S, F.CG)) continue;
      const cgPer = F.CG / S;
      for (const [P, Q] of opt.blocks) {
        if (!divides(Q, cgPer) || !qOk(Q) || P > T) continue;
        let wgTime = 0, work = 0;
        const stages = shared.map(n => {
          const r = R.get(n), lh = r.y1 - r.y0, lw = r.x1 - r.x0;
          const sq = [Q, 2, 1].find(q => q <= Q && divides(q, n.CG)), sp = Math.min(P, lw);
          const units = (n.CG / sq) * lh * ceil(lw, sp), c = cost(n, sp, sq);
          wgTime += ceil(units, opt.wg) * c + opt.barrierCost; work += units * c;
          return { n, P: sp, Q: sq };
        });
        const fUnits = (cgPer / Q) * T * ceil(T, P), fc = cost(F, P, Q);
        wgTime += ceil(fUnits, opt.wg) * fc; work += fUnits * fc;
        const nwg = tiles * S, total = work * nwg;
        if (total > base * opt.maxRecompute) continue;
        const time = ceil(nwg, Math.max(1, Math.floor(opt.concThreads / opt.wg))) * wgTime;
        if (!best || time < best.time) best = { T, S, P, Q, tiles, nwg, time, work: total, base, macs: order.reduce((a, n) => a + usefulMacs(n), 0), bytes, R, stages, mode, order };
      }
    }
  }
  return best;
}

export function planNet(net, opt = DEFAULTS) {
  // Start with one kernel per op; the input tensor is never a member.
  const groupOf = new Map();
  const groups = new Set();
  for (const n of net.order) {
    if (n.op === 'data') continue;
    const g = { members: new Set([n]), F: n };
    g.plan = planGroup(g.members, n, opt);
    if (!g.plan) throw new Error(`no feasible plan for ${n.name}`);
    groups.add(g); groupOf.set(n, g);
  }
  const score = g => g.plan.time + opt.dispatchCost;
  // Greedily pull producers into their consumer's kernel while the cost model says it pays.
  for (let changed = opt.fuse !== false; changed;) {
    changed = false;
    const list = [...groups].sort((a, b) => b.F.idx - a.F.idx);
    for (const g of list) {
      if (!groups.has(g)) continue;
      const prods = new Set();
      for (const m of g.members) for (const x of m.ins) if (!g.members.has(x) && groupOf.has(x)) prods.add(groupOf.get(x));
      let pick = null;
      for (const p of prods) {
        if (!p.F.cons.every(c => g.members.has(c))) continue;
        if (p.F.toGap) continue;
        const members = new Set([...g.members, ...p.members]);
        const plan = planGroup(members, g.F, opt);
        if (!plan) continue;
        const gain = score(g) + score(p) - (plan.time + opt.dispatchCost);
        if (gain > 0 && (!pick || gain > pick.gain)) pick = { p, members, plan, gain };
      }
      if (pick) {
        groups.delete(pick.p);
        g.members = pick.members; g.plan = pick.plan;
        for (const m of pick.p.members) groupOf.set(m, g);
        changed = true;
      }
    }
  }
  const list = [...groups].sort((a, b) => a.F.idx - b.F.idx);
  // Level = longest chain of kernels from the input; same-level kernels are independent.
  for (const g of list) {
    let lv = 0;
    for (const m of g.members) for (const x of m.ins) if (!g.members.has(x) && groupOf.has(x)) lv = Math.max(lv, groupOf.get(x).level + 1);
    g.level = lv;
  }
  return list;
}

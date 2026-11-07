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

export function evalCosts(members, mode, ext) {
  const memo = new Map();
  const read = n => !members.has(n) ? (ext?.(n) ?? 1) : mode.get(n) === 'shared' ? 1 : cost(n);
  const cost = n => {
    if (memo.has(n)) return memo.get(n);
    let c;
    if (n.op === 'conv') c = n.k[0] * n.k[1] * n.ins[0].CG * (4 + read(n.ins[0])) + 4;
    else if (n.op === 'dw') c = n.k[0] * n.k[1] * (1 + read(n.ins[0])) + 4;
    else if (n.op === 'add') c = read(n.ins[0]) + read(n.ins[1]) + 3;
    else if (n.op === 'concat') c = Math.max(...n.ins.map(read)) + 2;
    memo.set(n, c);
    return c;
  };
  return cost;
}

// Baseline: every op its own kernel, every tensor in global memory.
function soloWork(n) {
  const units = n.CG * n.H * n.W;
  if (n.op === 'conv') return units * (n.k[0] * n.k[1] * n.ins[0].CG * 5 + 4);
  if (n.op === 'dw') return units * (n.k[0] * n.k[1] * 2 + 4);
  return units * 4;
}

export function planGroup(members, F, opt, extCost) {
  const order = [...members].sort((a, b) => a.idx - b.idx);
  const mode = modes(members, F);
  const cost = evalCosts(members, mode, extCost);
  const shared = order.filter(n => mode.get(n) === 'shared');
  const base = order.reduce((a, n) => a + soloWork(n), 0);
  let best = null;
  for (const T of new Set([...opt.tiles.filter(t => t < F.H), F.H])) {
    if (T > F.H) continue;
    const R = regions(order, F, T);
    if (!R) return null;
    let bytes = F.toGap ? F.C * 4 : 0;
    const stages = [];
    for (const n of shared) {
      const r = R.get(n), lh = r.y1 - r.y0, lw = r.x1 - r.x0;
      bytes += n.CG * lh * lw * 16;
      stages.push({ n, r, units: n.CG * lh * lw });
    }
    if (bytes > opt.sharedBytes) continue;
    const tiles = ceil(F.H, T) * ceil(F.W, T);
    for (const S of opt.splits) {
      if (F.CG % S) continue;
      const fUnits = (F.CG / S) * T * T;
      let wgTime = 0, work = 0;
      for (const s of stages) { wgTime += ceil(s.units, opt.wg) * cost(s.n) + opt.barrierCost; work += s.units * cost(s.n); }
      wgTime += ceil(fUnits, opt.wg) * cost(F);
      work += fUnits * cost(F);
      const nwg = tiles * S, total = work * nwg;
      if (total > base * opt.maxRecompute) continue;
      const time = ceil(nwg, Math.max(1, Math.floor(opt.concThreads / opt.wg))) * wgTime;
      if (!best || time < best.time) best = { T, S, tiles, nwg, time, work: total, base, bytes, R, stages: stages.map(s => s.n), mode, order };
    }
  }
  return best;
}

export function planNet(net, opt = DEFAULTS, extCost) {
  // Start with one kernel per op; the input tensor is never a member.
  const groupOf = new Map();
  const groups = new Set();
  for (const n of net.order) {
    if (n.op === 'data') continue;
    const g = { members: new Set([n]), F: n };
    g.plan = planGroup(g.members, n, opt, extCost);
    if (!g.plan) throw new Error(`no feasible plan for ${n.name}`);
    groups.add(g); groupOf.set(n, g);
  }
  const score = g => g.plan.time + opt.dispatchCost;
  // Greedily pull producers into their consumer's kernel while the cost model says it pays.
  for (let changed = true; changed;) {
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
        const plan = planGroup(members, g.F, opt, extCost);
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

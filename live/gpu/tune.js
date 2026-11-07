// Measurement-driven autotuner. Times candidate kernels on the real GPU (many back-to-back dispatches between
// two timestamps), then makes the same greedy fusion decisions as the planner, but on measured times.
//
// quick: the cheap on-device variant. Single-op kernels only (no fusion search), a few configs per op chosen for
// variety (one per schedule x register-block shape), results shared between ops of identical shape, and wall-clock
// timing when the GPU has no timestamp queries. Seconds on a desktop GPU instead of minutes. With a seed (a full tuning
// file from another GPU) quick mode keeps its fusion groups and only re-picks each group's tile/split/block/schedule;
// groups that can't run here (e.g. not enough workgroup memory) fall back to single ops.
import { parseNet, packWeights, parseFloatNet, packFloatWeights } from './graph.js';
import { candidates, DEFAULTS } from './fuse.js';
import { moduleSource } from './codegen.js';

export async function tune(device, specs, { opt: over = {}, quick = false, seed = null, signal = null, topK = quick ? 4 : 24, reps = quick ? 16 : 40, overheadUs = 2, log = () => {}, onProgress = () => {} } = {}) {
  const stamps = device.features.has('timestamp-query');
  if (!stamps && !quick) throw new Error('full tuning needs timestamp-query');
  const opt = { ...DEFAULTS, sharedBytes: device.limits.maxComputeWorkgroupStorageSize, maxRecompute: 4, ...over };
  const nets = specs.map(s => s.dtype === 'f32' ? parseFloatNet(s.ir, s.wanted) : parseNet(s.spec, s.wanted));
  const packed = nets.map((n, i) => n.dtype === 'f32' ? packFloatWeights(n, specs[i].buf) : packWeights(n, specs[i].buf, { u24: opt.u24 !== false }));
  const wBase = [];
  let wLen = 0;
  for (const p of packed) { wBase.push(wLen); wLen += p.length / 4; }

  const U = GPUBufferUsage;
  const wts = device.createBuffer({ size: wLen * 16, usage: U.STORAGE | U.COPY_DST });
  packed.forEach((p, i) => device.queue.writeBuffer(wts, wBase[i] * 16, p));
  const arenaVec = 1 << 20; // 16 MiB scratch: inputs then output, batch 1
  const act = device.createBuffer({ size: arenaVec * 16, usage: U.STORAGE });
  const sums = device.createBuffer({ size: 1 << 16, usage: U.STORAGE });
  const cfg = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(cfg, 0, new Int32Array([1, 1, 1, 1]));
  const bgl = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(b => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: b === 0 ? 'read-only-storage' : b === 3 ? 'uniform' : 'storage' } })) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const bg = device.createBindGroup({ layout: bgl, entries: [wts, act, sums, cfg].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const qs = stamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const qRes = stamps ? device.createBuffer({ size: 16, usage: U.QUERY_RESOLVE | U.COPY_SRC }) : null;
  const qRead = stamps ? device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST }) : null;

  async function time(pipeline, nwg) {
    const best = [];
    for (let r = 0; r < 3; r++) {
      if (!stamps) {
        const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bg);
        for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(nwg);
        pass.end();
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        best.push((performance.now() - t0) * 1e3 / reps);
        continue;
      }
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
      pass.setPipeline(pipeline); pass.setBindGroup(0, bg);
      for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(nwg);
      pass.end();
      enc.resolveQuerySet(qs, 0, 2, qRes, 0);
      enc.copyBufferToBuffer(qRes, 0, qRead, 0, 16);
      device.queue.submit([enc.finish()]);
      await qRead.mapAsync(GPUMapMode.READ);
      const t = new BigInt64Array(qRead.getMappedRange().slice(0));
      qRead.unmap();
      best.push(Number(t[1] - t[0]) / 1e3 / reps);
    }
    return Math.min(...best);
  }

  const memo = new Map();
  let measured = 0;
  // Ops with the same shape and parameters time the same; quick mode measures each shape once.
  const shapeKey = (ni, n) => JSON.stringify([nets[ni].dtype, n.op, n.C, n.H, n.W, n.k, n.s, n.p, n.act, !!n.u24, n.toGap, n.ins.map(x => [x.op === 'data', x.C, x.H, x.W])]);
  // A fused group's key: each member's shape plus how members connect (inside the group or from outside).
  const groupKey = (ni, members) => {
    const list = [...members].sort((a, b) => a.idx - b.idx), pos = new Map(list.map((m, i) => [m, i]));
    return JSON.stringify(list.map(m => [shapeKey(ni, m), m.ins.map(x => pos.has(x) ? pos.get(x) : -1)]));
  };
  const variety = list => {
    const seen = new Set(), out = [];
    for (const c of list) { const k = `${c.sched}:${c.P}x${c.Q}`; if (!seen.has(k)) { seen.add(k); out.push(c); } if (out.length >= topK) break; }
    return out;
  };
  // Best measured config for a group; null if nothing compiles or fits.
  async function measureGroup(ni, members, F) {
    signal?.throwIfAborted();
    const key = quick ? groupKey(ni, members) : `${ni}:${F.name}:${[...members].map(m => m.idx).sort((a, b) => a - b).join(',')}`;
    if (memo.has(key)) return memo.get(key);
    const all = candidates(members, F, opt), cands = quick ? variety(all) : all.slice(0, topK);
    // Scratch placement: every external input, then the output.
    let off = 0;
    const ext = new Set();
    for (const m of members) for (const x of m.ins) if (!members.has(x)) ext.add(x);
    for (const x of [...ext, F]) { x.aOff = off; off += x.CG * x.H * x.W; }
    if (off > arenaVec) throw new Error('scratch arena too small');
    F.sumOff = 0;
    const ctx = { wg: opt.wg, source: 'tensor', nets: nets.length };
    const built = await Promise.all(cands.map(async plan => {
      const k = { group: { members, F, plan }, net: ni, wBase: wBase[ni], dtype: nets[ni].dtype };
      const src = moduleSource([k], ctx);
      device.pushErrorScope('validation');
      try {
        const module = device.createShaderModule({ code: src.code });
        const pipeline = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
        const err = await device.popErrorScope();
        return err ? null : { plan, pipeline, nwg: src.perNet.reduce((a, b) => a + b, 0) };
      } catch (e) { await device.popErrorScope(); return null; }
    }));
    let best = null;
    for (const b of built) {
      if (!b) continue;
      const us = await time(b.pipeline, b.nwg);
      measured++;
      if (!best || us < best.us) best = { plan: b.plan, us };
    }
    memo.set(key, best);
    if (quick) onProgress(memo.size / totalShapes);
    return best;
  }
  // Seeded groups per net (member sets), else single ops; used for quick mode's progress estimate and its work list.
  const seededGroups = (net, ni) => {
    const want = seed?.nets?.[net.name];
    if (!want) return net.order.filter(n => n.op !== 'data').map(n => ({ F: n, members: new Set([n]) }));
    const covered = new Set(want.flatMap(g => g.members));
    const gs = want.filter(g => net.nodes.has(g.F)).map(g => ({ F: net.nodes.get(g.F), members: new Set(g.members.map(m => net.nodes.get(m)).filter(Boolean)) }));
    for (const n of net.order) if (n.op !== 'data' && !covered.has(n.name)) gs.push({ F: n, members: new Set([n]) });
    return gs;
  };
  const totalShapes = quick ? new Set(nets.flatMap((net, ni) => seededGroups(net, ni).map(g => groupKey(ni, g.members)))).size : 0;

  const result = { opt: { wg: opt.wg, u24: opt.u24 !== false }, nets: {}, totalUs: 0 };
  for (let ni = 0; ni < nets.length; ni++) {
    const net = nets[ni], groupOf = new Map(), groups = new Set();
    const add = async (members, F) => {
      const m = await measureGroup(ni, members, F);
      if (!m) {
        if (members.size === 1) throw new Error(`nothing runs for ${F.name}`);
        for (const n of members) await add(new Set([n]), n); // seeded group doesn't fit this GPU: split it
        return;
      }
      const g = { members, F, ...m };
      groups.add(g);
      for (const n of members) groupOf.set(n, g);
    };
    for (const g of quick ? seededGroups(net, ni) : net.order.filter(n => n.op !== 'data').map(n => ({ F: n, members: new Set([n]) }))) await add(g.members, g.F);
    log(`${net.name}: ${groups.size} single-op kernels, ${[...groups].reduce((a, g) => a + g.us, 0).toFixed(0)} us`);
    for (let round = 1, changed = !quick; changed; round++) {
      changed = false;
      for (const g of [...groups].sort((a, b) => b.F.idx - a.F.idx)) {
        if (!groups.has(g)) continue;
        const prods = new Set();
        for (const m of g.members) for (const x of m.ins) if (!g.members.has(x) && groupOf.has(x)) prods.add(groupOf.get(x));
        let pick = null;
        for (const p of prods) {
          if (p.F.toGap || !p.F.cons.every(c => g.members.has(c))) continue;
          const members = new Set([...g.members, ...p.members]);
          const m = await measureGroup(ni, members, g.F);
          if (!m) continue;
          const gain = g.us + p.us + overheadUs - m.us;
          if (gain > 0 && (!pick || gain > pick.gain)) pick = { p, members, m, gain };
        }
        if (pick) {
          groups.delete(pick.p);
          g.members = pick.members; g.plan = pick.m.plan; g.us = pick.m.us;
          for (const m of pick.p.members) groupOf.set(m, g);
          changed = true;
        }
      }
      log(`${net.name} round ${round}: ${groups.size} kernels, ${[...groups].reduce((a, g) => a + g.us, 0).toFixed(0)} us (${measured} configs timed)`);
    }
    const list = [...groups].sort((a, b) => a.F.idx - b.F.idx);
    result.nets[net.name] = list.map(g => ({ F: g.F.name, members: [...g.members].sort((a, b) => a.idx - b.idx).map(m => m.name), T: g.plan.T, S: g.plan.S, P: g.plan.P, Q: g.plan.Q, sched: g.plan.sched, us: +g.us.toFixed(2) }));
    result.totalUs += list.reduce((a, g) => a + g.us, 0);
  }
  [wts, act, sums, cfg, qRes, qRead].forEach(b => b?.destroy());
  result.quick = quick;
  result.timer = stamps ? 'timestamp' : 'wall';
  return result;
}

// Measurement-driven autotuner. Times candidate kernels on the real GPU (many back-to-back dispatches between
// two timestamps), then makes the same greedy fusion decisions as the planner, but on measured times.
import { parseNet, packWeights } from './graph.js';
import { candidates, DEFAULTS } from './fuse.js';
import { moduleSource } from './codegen.js';

export async function tune(device, specs, { opt: over = {}, topK = 24, reps = 40, overheadUs = 2, log = () => {} } = {}) {
  if (!device.features.has('timestamp-query')) throw new Error('tuning needs timestamp-query');
  const opt = { ...DEFAULTS, sharedBytes: device.limits.maxComputeWorkgroupStorageSize, maxRecompute: 4, ...over };
  const nets = specs.map(s => parseNet(s.spec, s.wanted));
  const packed = nets.map((n, i) => packWeights(n, specs[i].buf, { u24: opt.u24 !== false }));
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
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
  const qRes = device.createBuffer({ size: 16, usage: U.QUERY_RESOLVE | U.COPY_SRC });
  const qRead = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });

  async function time(pipeline, nwg) {
    const best = [];
    for (let r = 0; r < 3; r++) {
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
  // Best measured config for a group; null if nothing compiles or fits.
  async function measureGroup(ni, members, F) {
    const key = `${ni}:${F.name}:${[...members].map(m => m.idx).sort((a, b) => a - b).join(',')}`;
    if (memo.has(key)) return memo.get(key);
    const cands = candidates(members, F, opt).slice(0, topK);
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
    return best;
  }

  const result = { opt: { wg: opt.wg, u24: opt.u24 !== false }, nets: {}, totalUs: 0 };
  for (let ni = 0; ni < nets.length; ni++) {
    const net = nets[ni], groupOf = new Map(), groups = new Set();
    for (const n of net.order) {
      if (n.op === 'data') continue;
      const members = new Set([n]), m = await measureGroup(ni, members, n);
      if (!m) throw new Error(`nothing runs for ${n.name}`);
      const g = { members, F: n, ...m };
      groups.add(g); groupOf.set(n, g);
    }
    log(`${net.name}: ${groups.size} single-op kernels, ${[...groups].reduce((a, g) => a + g.us, 0).toFixed(0)} us`);
    for (let round = 1, changed = true; changed; round++) {
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
  [wts, act, sums, cfg, qRes, qRead].forEach(b => b.destroy());
  return result;
}

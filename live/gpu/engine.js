// WebGPU runtime: plans and compiles both nets into one sequence of fused dispatches over a shared
// activation arena, runs a batch of faces per submit, and finishes the tiny fc/softmax tail on the CPU.
import { parseNet, packWeights, runTail } from './graph.js';
import { planNet, DEFAULTS } from './fuse.js';
import { moduleSource } from './codegen.js';

export async function requestDevice() {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  const features = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    },
  });
  return { adapter, device, info: adapter.info };
}

// First-fit arena allocation over [first, last] dispatch lifetimes, in vec4 units.
function allocate(tensors) {
  const placed = [];
  let top = 0;
  for (const t of [...tensors].sort((a, b) => b.size - a.size)) {
    const clash = placed.filter(p => !(p.last < t.first || t.last < p.first)).sort((a, b) => a.off - b.off);
    let off = 0;
    for (const p of clash) { if (off + t.size <= p.off) break; off = Math.max(off, p.off + p.size); }
    t.off = off; placed.push(t); top = Math.max(top, off + t.size);
  }
  return top;
}

export async function createEngine(device, specs, { maxBatch = 4, source = 'tensor', opt = {}, profile = false } = {}) {
  const o = { ...DEFAULTS, sharedBytes: device.limits.maxComputeWorkgroupStorageSize, ...opt };
  const nets = specs.map(s => parseNet(s.spec, s.wanted));

  // Weights for both nets in one buffer.
  const packed = nets.map((n, i) => packWeights(n, specs[i].buf));
  const wBase = [];
  let wLen = 0;
  for (const p of packed) { wBase.push(wLen); wLen += p.length / 4; }

  const groups = nets.map(n => planNet(n, o));
  const levels = Math.max(...groups.map(g => Math.max(...g.map(x => x.level)))) + 1;

  // Lifetimes: a tensor lives from the dispatch that writes it to the last one that reads it.
  const tensors = [];
  nets.forEach((net, ni) => {
    const owner = new Map();
    for (const g of groups[ni]) for (const m of g.members) owner.set(m, g);
    const lastUse = n => Math.max(...n.cons.map(c => owner.get(c).level));
    if (source === 'tensor') tensors.push({ node: net.data, first: -1, last: lastUse(net.data), size: maxBatch * net.data.CG * net.data.H * net.data.W });
    for (const g of groups[ni]) if (!g.F.toGap) tensors.push({ node: g.F, first: g.level, last: lastUse(g.F), size: maxBatch * g.F.CG * g.F.H * g.F.W });
  });
  const arenaVec = allocate(tensors);
  for (const t of tensors) t.node.aOff = t.off;

  let sumLen = 0;
  for (const net of nets) for (const g of net.gaps) {
    const src = net.nodes.get(g.src);
    src.sumOff = g.off = sumLen;
    sumLen += maxBatch * src.C;
  }

  // One module per level holding every kernel of that level from both nets.
  const ctx = { wg: o.wg, source, nets: nets.length };
  const stages = [];
  for (let l = 0; l < levels; l++) {
    const kernels = [];
    groups.forEach((gs, ni) => gs.filter(g => g.level === l).forEach(group => kernels.push({ group, net: ni, wBase: wBase[ni] })));
    if (o.hfuse === false) for (const k of kernels) stages.push({ level: l, kernels: [k], ...moduleSource([k], ctx) });
    else if (kernels.length) stages.push({ level: l, kernels, ...moduleSource(kernels, ctx) });
  }

  const entries = [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ];
  if (source === 'texture') entries.push(
    { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
  const bgl = device.createBindGroupLayout({ entries });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const t0 = performance.now();
  device.pushErrorScope('validation');
  await Promise.all(stages.map(async s => {
    const module = device.createShaderModule({ code: s.code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error(`WGSL error in level ${s.level}: ${errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
    s.pipeline = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
  }));
  const vErr = await device.popErrorScope();
  if (vErr) throw new Error(vErr.message);
  const compileMs = performance.now() - t0;

  const buf = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
  const wts = buf(wLen * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  packed.forEach((p, i) => device.queue.writeBuffer(wts, wBase[i] * 16, p));
  const act = buf(arenaVec * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const sums = buf(sumLen * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const cfg = buf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const crops = buf(maxBatch * nets.length * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  // A small ring of readback buffers so a new frame can be submitted while older results are mapping.
  const ring = Array.from({ length: 3 }, () => ({ buf: buf(sumLen * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST), busy: false }));

  const canTime = profile && device.features.has('timestamp-query');
  const qs = canTime ? device.createQuerySet({ type: 'timestamp', count: stages.length * 2 }) : null;
  const qRes = canTime ? buf(stages.length * 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC) : null;
  const qRead = canTime ? buf(stages.length * 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : null;

  let bindGroup = null, boundView = null;
  const bind = view => {
    if (bindGroup && boundView === view) return bindGroup;
    const e = [{ binding: 0, resource: { buffer: wts } }, { binding: 1, resource: { buffer: act } }, { binding: 2, resource: { buffer: sums } }, { binding: 3, resource: { buffer: cfg } }];
    if (source === 'texture') e.push({ binding: 4, resource: view }, { binding: 5, resource: { buffer: crops } });
    bindGroup = device.createBindGroup({ layout: bgl, entries: e });
    boundView = view;
    return bindGroup;
  };

  const cfgData = new Int32Array(4);
  function encode(enc, B, view) {
    cfgData[0] = B;
    device.queue.writeBuffer(cfg, 0, cfgData);
    enc.clearBuffer(sums);
    const bg = bind(view);
    if (canTime) {
      stages.forEach((s, i) => {
        const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        pass.setPipeline(s.pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(B * s.wgPerItem); pass.end();
      });
      enc.resolveQuerySet(qs, 0, stages.length * 2, qRes, 0);
      enc.copyBufferToBuffer(qRes, 0, qRead, 0, stages.length * 16);
    } else {
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, bg);
      for (const s of stages) { pass.setPipeline(s.pipeline); pass.dispatchWorkgroups(B * s.wgPerItem); }
      pass.end();
    }
  }

  async function finish(slot, B) {
    await slot.buf.mapAsync(GPUMapMode.READ, 0, sumLen * 4);
    const s = new Int32Array(slot.buf.getMappedRange(0, sumLen * 4).slice(0));
    slot.buf.unmap(); slot.busy = false;
    let times = null;
    if (canTime) {
      await qRead.mapAsync(GPUMapMode.READ);
      const t = new BigInt64Array(qRead.getMappedRange().slice(0));
      qRead.unmap();
      times = stages.map((st, i) => Number(t[2 * i + 1] - t[2 * i]) / 1e3);
    }
    const res = [];
    for (let b = 0; b < B; b++) res.push(nets.map((net, i) => runTail(net, specs[i].buf, s, b)));
    return { res, times };
  }

  function submit(B, view, before) {
    if (B < 1 || B > maxBatch) throw new Error(`batch ${B} out of range`);
    const slot = ring.find(r => !r.busy);
    if (!slot) return null; // GPU is behind; caller drops this frame
    slot.busy = true;
    const enc = device.createCommandEncoder();
    before?.(enc);
    encode(enc, B, view);
    enc.copyBufferToBuffer(sums, 0, slot.buf, 0, sumLen * 4);
    device.queue.submit([enc.finish()]);
    return finish(slot, B);
  }

  // Test path: list[b][net] is the reference HWC Int16 tensor.
  function runTensors(list) {
    if (source !== 'tensor') throw new Error('engine built for texture input');
    nets.forEach((net, i) => {
      const d = net.data, n = d.H * d.W, a = new Int32Array(list.length * d.CG * n * 4);
      list.forEach((items, b) => {
        const t = items[i];
        for (let p = 0; p < n; p++) for (let c = 0; c < d.C; c++) a[(b * d.CG * n + p) * 4 + c] = t[p * d.C + c];
      });
      device.queue.writeBuffer(act, d.aOff * 16, a);
    });
    return submit(list.length, null);
  }

  // Live path: crops[b][net] = {m:[a,b,c,d], t:[e,f], lvl} mapping crop pixel centers to frame texels.
  const cropData = new ArrayBuffer(maxBatch * nets.length * 32), cf = new Float32Array(cropData), ci = new Int32Array(cropData);
  function runFrame(view, list, before) {
    list.forEach((items, b) => items.forEach((c, i) => {
      const k = (b * nets.length + i) * 8;
      cf.set(c.m, k); cf.set(c.t, k + 4); ci[k + 6] = c.lvl; ci[k + 7] = 0;
    }));
    device.queue.writeBuffer(crops, 0, cropData, 0, list.length * nets.length * 32);
    return submit(list.length, view, before);
  }

  const kernelCount = groups.reduce((a, g) => a + g.length, 0);
  return {
    runTensors, runFrame, nets, stages, compileMs, kernelCount, dispatches: stages.length,
    arenaBytes: arenaVec * 16, weightBytes: wLen * 16, canTime, idle: () => ring.some(r => !r.busy),
    destroy() { [wts, act, sums, cfg, crops, ...ring.map(r => r.buf)].forEach(b => b.destroy()); },
  };
}

export function expScores(T) {
  return { attractiveRaw: T.predict_attractive.v[0], attractive: Math.min(100, Math.max(0, T.predict_attractive.v[0] + 15)), valence: T.predict_valence.v[0], arousal: T.predict_arousal.v[0], exp: Array.from(T.prob_exp.v) };
}
export function ageScores(T) { return { age: T.predict_age.v[0] * 10, boy: T.prob_gender.v[1] }; }

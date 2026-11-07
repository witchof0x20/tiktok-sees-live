// WebGPU runtime. createProgram plans and compiles a set of nets (fixed-point or float) into one sequence of fused
// dispatches over a shared activation arena; it only encodes, so callers can chain programs and their own kernels in
// one submit. Batches are per net: either written by the CPU (uniform) or by an earlier GPU kernel (storage), in which
// case dispatch sizes come from an indirect-args buffer the program fills itself. createEngine wraps a program with
// uploads, readback and the CPU tail for the fixed-point nets.
import { parseNet, packWeights, runTail, parseFloatNet, packFloatWeights } from './graph.js';
import { planNet, planFromTuned, DEFAULTS } from './fuse.js';
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

const U = () => GPUBufferUsage;

// specs[i]: { spec, buf, wanted } for a fixed-point net, or { ir, buf, wanted, dtype: 'f32' } for a float net.
export async function createProgram(device, specs, { maxBatch = 4, source = 'tensor', opt = {}, profile = false, tuned = null, batchBuffer = null, cropsBuffer = null, label = '' } = {}) {
  if (specs.length > 4) throw new Error('at most 4 nets per program');
  const o = { ...DEFAULTS, sharedBytes: device.limits.maxComputeWorkgroupStorageSize, ...opt };
  const nets = specs.map(s => s.dtype === 'f32' ? parseFloatNet(s.ir, s.wanted) : parseNet(s.spec, s.wanted));
  const packed = nets.map((n, i) => n.dtype === 'f32' ? packFloatWeights(n, specs[i].buf) : packWeights(n, specs[i].buf, { u24: (tuned?.opt?.u24 ?? o.u24) !== false }));
  const wBase = [];
  let wLen = 0;
  for (const p of packed) { wBase.push(wLen); wLen += p.length / 4; }

  const groups = nets.map(n => tuned?.nets?.[n.name] ? planFromTuned(n, tuned.nets[n.name], { ...o, ...tuned.opt, sharedBytes: o.sharedBytes }) : planNet(n, o));
  const levels = Math.max(...groups.map(g => Math.max(...g.map(x => x.level)))) + 1;

  // Lifetimes: a tensor lives from the dispatch that writes it to the last one that reads it; net outputs live on.
  const tensors = [];
  nets.forEach((net, ni) => {
    const owner = new Map();
    for (const g of groups[ni]) for (const m of g.members) owner.set(m, g);
    const outs = new Set(net.outputs || []);
    const lastUse = n => outs.has(n) ? levels + 1 : Math.max(...n.cons.map(c => owner.get(c).level));
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

  const ctx = { wg: o.wg, source, nets: nets.length, gpuBatch: !!batchBuffer };
  const stages = [];
  for (let l = 0; l < levels; l++) {
    const kernels = [];
    groups.forEach((gs, ni) => gs.filter(g => g.level === l).forEach(group => kernels.push({ group, net: ni, wBase: wBase[ni], dtype: nets[ni].dtype })));
    if (o.hfuse === false) for (const k of kernels) stages.push({ level: l, kernels: [k], ...moduleSource([k], ctx) });
    else if (kernels.length) stages.push({ level: l, kernels, ...moduleSource(kernels, ctx) });
  }

  const entries = [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: batchBuffer ? 'read-only-storage' : 'uniform' } },
  ];
  if (source === 'texture') entries.push(
    { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
  const bgl = device.createBindGroupLayout({ entries });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  // GPU-driven batches: one tiny kernel turns the batch vector into indirect dispatch args for every stage.
  let argsPipeline = null;
  if (batchBuffer) {
    const lines = stages.map((s, i) => `  a[${3 * i}] = u32(${s.perNet.map((n, ni) => n ? `max(cfg.batch[${ni}], 0) * ${n}` : '').filter(Boolean).join(' + ') || '0'}); a[${3 * i + 1}] = 1u; a[${3 * i + 2}] = 1u;`);
    const code = `struct Cfg { batch: vec4<i32> }
@group(0) @binding(0) var<storage, read> cfg: Cfg;
@group(0) @binding(1) var<storage, read_write> a: array<u32>;
@compute @workgroup_size(1) fn main() {
${lines.join('\n')}
}`;
    const module = device.createShaderModule({ code, label: `${label} args` });
    const errs = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error(`WGSL error in ${label} args: ${errs.map(m => m.message).join('; ')}`);
    argsPipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }

  const t0 = performance.now();
  device.pushErrorScope('validation');
  await Promise.all(stages.map(async s => {
    const module = device.createShaderModule({ code: s.code, label: `${label} level ${s.level}` });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error(`WGSL error in ${label} level ${s.level}: ${errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
    s.pipeline = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
  }));
  const vErr = await device.popErrorScope();
  if (vErr) throw new Error(vErr.message);
  const compileMs = performance.now() - t0;

  const buf = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
  const wts = buf(wLen * 16, U().STORAGE | U().COPY_DST);
  packed.forEach((p, i) => device.queue.writeBuffer(wts, wBase[i] * 16, p));
  const act = buf(arenaVec * 16, U().STORAGE | U().COPY_DST | U().COPY_SRC);
  const sums = buf(sumLen * 4, U().STORAGE | U().COPY_SRC | U().COPY_DST);
  const cfg = batchBuffer || buf(16, U().UNIFORM | U().COPY_DST);
  const crops = cropsBuffer || buf(maxBatch * nets.length * 32, U().STORAGE | U().COPY_DST);
  const args = batchBuffer ? buf(stages.length * 12, U().STORAGE | U().INDIRECT) : null;
  const argsBind = batchBuffer ? device.createBindGroup({ layout: argsPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: batchBuffer } }, { binding: 1, resource: { buffer: args } }] }) : null;

  const canTime = profile && device.features.has('timestamp-query');
  const qs = canTime ? device.createQuerySet({ type: 'timestamp', count: stages.length * 2 }) : null;
  const qRes = canTime ? buf(stages.length * 16, U().QUERY_RESOLVE | U().COPY_SRC) : null;
  const qRead = canTime ? buf(stages.length * 16, U().MAP_READ | U().COPY_DST) : null;

  const bindCache = new Map();
  const bind = view => {
    if (bindCache.has(view)) return bindCache.get(view);
    if (bindCache.size > 16) bindCache.clear();
    const e = [{ binding: 0, resource: { buffer: wts } }, { binding: 1, resource: { buffer: act } }, { binding: 2, resource: { buffer: sums } }, { binding: 3, resource: { buffer: cfg } }];
    if (source === 'texture') e.push({ binding: 4, resource: view }, { binding: 5, resource: { buffer: crops } });
    const bindGroup = device.createBindGroup({ layout: bgl, entries: e });
    bindCache.set(view, bindGroup);
    return bindGroup;
  };

  const cfgData = new Int32Array(4);
  // batch: per-net array (CPU batches) — ignored when batches come from the GPU.
  function encode(enc, { view = null, batch = null } = {}) {
    if (!batchBuffer) {
      cfgData.fill(0);
      batch.forEach((b, i) => { if (b < 0 || b > maxBatch) throw new Error(`batch ${b} out of range`); cfgData[i] = b; });
      device.queue.writeBuffer(cfg, 0, cfgData);
    }
    if (sumLen) enc.clearBuffer(sums);
    const bg = bind(view);
    const size = s => s.perNet.reduce((a, n, ni) => a + n * (cfgData[ni] || 0), 0);
    const run = (pass, s, i) => {
      pass.setPipeline(s.pipeline); pass.setBindGroup(0, bg);
      if (batchBuffer) pass.dispatchWorkgroupsIndirect(args, i * 12); else pass.dispatchWorkgroups(size(s));
    };
    if (batchBuffer) {
      const pass = enc.beginComputePass();
      pass.setPipeline(argsPipeline); pass.setBindGroup(0, argsBind); pass.dispatchWorkgroups(1); pass.end();
    }
    if (canTime) {
      stages.forEach((s, i) => {
        const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        run(pass, s, i); pass.end();
      });
      enc.resolveQuerySet(qs, 0, stages.length * 2, qRes, 0);
      enc.copyBufferToBuffer(qRes, 0, qRead, 0, stages.length * 16);
    } else {
      const pass = enc.beginComputePass();
      stages.forEach((s, i) => run(pass, s, i));
      pass.end();
    }
  }
  async function readTimes() {
    if (!canTime) return null;
    await qRead.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(qRead.getMappedRange().slice(0));
    qRead.unmap();
    return stages.map((st, i) => Number(t[2 * i + 1] - t[2 * i]) / 1e3);
  }
  // Where a net's output tensor lives in the arena (vec4 units), for downstream kernels and readback.
  const output = (ni, name) => { const n = nets[ni].nodes.get(name); if (n?.aOff == null) throw new Error(`${name} is not a materialized output`); return { off: n.aOff, CG: n.CG, H: n.H, W: n.W, C: n.C }; };
  // Uploads reference HWC tensors (Int16Array for fixed-point, Float32Array for float) as net inputs.
  function upload(list) {
    nets.forEach((net, i) => {
      const d = net.data, n = d.H * d.W, rows = list.map(items => items[i]).filter(Boolean);
      const a = net.dtype === 'f32' ? new Float32Array(rows.length * d.CG * n * 4) : new Int32Array(rows.length * d.CG * n * 4);
      rows.forEach((t, b) => { for (let p = 0; p < n; p++) for (let c = 0; c < d.C; c++) a[(b * d.CG * n + p) * 4 + c] = t[p * d.C + c]; });
      if (rows.length) device.queue.writeBuffer(act, d.aOff * 16, a);
    });
  }

  return {
    encode, readTimes, output, upload, nets, specs, stages, compileMs, act, sums, sumLen, crops, maxBatch, canTime,
    kernelCount: groups.reduce((a, g) => a + g.length, 0), dispatches: stages.length, arenaBytes: arenaVec * 16, weightBytes: wLen * 16,
    destroy() { [wts, act, sums, args, qRes, qRead, ...(batchBuffer ? [] : [cfg]), ...(cropsBuffer ? [] : [crops])].forEach(b => b?.destroy()); },
  };
}

// The fixed-point TikTok nets with their own uploads and readback: one submit per batch of faces.
export async function createEngine(device, specs, options = {}) {
  const prog = await createProgram(device, specs, options);
  const { nets, sumLen, maxBatch } = prog;
  const ring = Array.from({ length: 3 }, () => ({ buf: device.createBuffer({ size: Math.max(16, sumLen * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));

  async function finish(slot, B) {
    await slot.buf.mapAsync(GPUMapMode.READ, 0, sumLen * 4);
    const s = new Int32Array(slot.buf.getMappedRange(0, sumLen * 4).slice(0));
    slot.buf.unmap(); slot.busy = false;
    const times = await prog.readTimes();
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
    prog.encode(enc, { view, batch: nets.map(() => B) });
    enc.copyBufferToBuffer(prog.sums, 0, slot.buf, 0, sumLen * 4);
    device.queue.submit([enc.finish()]);
    return finish(slot, B);
  }
  function runTensors(list) { prog.upload(list); return submit(list.length, null); }
  const cropData = new ArrayBuffer(maxBatch * nets.length * 32), cf = new Float32Array(cropData), ci = new Int32Array(cropData);
  function runFrame(view, list, before) {
    writeCrops(cf, ci, list, nets.length);
    device.queue.writeBuffer(prog.crops, 0, cropData, 0, list.length * nets.length * 32);
    return submit(list.length, view, before);
  }
  return {
    ...prog, runTensors, runFrame, idle: () => ring.some(r => !r.busy),
    destroy() { prog.destroy(); ring.forEach(r => r.buf.destroy()); },
  };
}

// crops[b][net] = {m:[a,b,c,d], t:[e,f], lvl}: crop pixel center -> frame texel, in the Crop struct layout.
export function writeCrops(cf, ci, list, nNets) {
  list.forEach((items, b) => items.forEach((c, i) => {
    const k = (b * nNets + i) * 8;
    cf.set(c.m, k); cf.set(c.t, k + 4); ci[k + 6] = c.lvl; ci[k + 7] = 0;
  }));
}

export function expScores(T) {
  return { attractiveRaw: T.predict_attractive.v[0], attractive: Math.min(100, Math.max(0, T.predict_attractive.v[0] + 15)), valence: T.predict_valence.v[0], arousal: T.predict_arousal.v[0], exp: Array.from(T.prob_exp.v) };
}
export function ageScores(T) { return { age: T.predict_age.v[0] * 10, boy: T.prob_gender.v[1] }; }

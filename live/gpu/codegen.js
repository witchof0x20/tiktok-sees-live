// WGSL generator. Every tensor is NC4HW4 (vec4<i32> per 4 channels per pixel). Each kernel is a set of
// per-node functions: global reads (zero outside the image, which is how padding works), workgroup-memory
// reads for staged nodes, and eval functions that compute a node from its inputs. Several kernels can be
// emitted into one module and picked by workgroup index (horizontal fusion).

// x >> s with round-half-up, as fixed-point rescale; s < 0 shifts left.
function rs(e, s) {
  if (s === 0) return e;
  if (s > 0) return `((${e} + vec4<i32>(${1 << (s - 1)})) >> vec4<u32>(${s}u))`;
  return `(${e} * vec4<i32>(${1 << -s}))`;
}
const sat = e => `clamp(${e}, vec4<i32>(-32768), vec4<i32>(32767))`;
const relu = e => `max(${e}, vec4<i32>(0))`;

function kernelSource(k, j, ctx) {
  const { members, F, plan } = k.group, { mode, R, T, S } = plan;
  const p = `k${j}_`, fn = n => `${p}n${n.idx}`, ev = n => `${p}e${n.idx}`;
  const out = [], seen = new Set();
  const wB = k.wBase;

  const emitRead = n => {
    if (seen.has(n)) return;
    seen.add(n);
    if (!members.has(n)) {
      if (n.op === 'data' && ctx.source === 'texture') {
        out.push(`fn ${fn(n)}(cg: i32, y: i32, x: i32) -> vec4<i32> {
  if (y < 0 || y >= ${n.H} || x < 0 || x >= ${n.W}) { return vec4<i32>(0); }
  return crop_px(b * ${ctx.nets} + ${k.net}, y, x);
}`);
      } else {
        out.push(`fn ${fn(n)}(cg: i32, y: i32, x: i32) -> vec4<i32> {
  if (y < 0 || y >= ${n.H} || x < 0 || x >= ${n.W}) { return vec4<i32>(0); }
  return act[${n.aOff} + ((b * ${n.CG} + cg) * ${n.H} + y) * ${n.W} + x];
}`);
      }
      return;
    }
    for (const x of n.ins) emitRead(x);
    emitEval(n);
    if (mode.get(n) === 'shared') {
      const r = R.get(n), lh = r.y1 - r.y0, lw = r.x1 - r.x0;
      out.push(`fn ${fn(n)}(cg: i32, y: i32, x: i32) -> vec4<i32> {
  return sh[${k.shOff.get(n)} + (cg * ${lh} + (y - (ty0 * ${r.scale} + ${r.y0}))) * ${lw} + (x - (tx0 * ${r.scale} + ${r.x0}))];
}`);
    } else if (mode.get(n) === 'inline') {
      out.push(`fn ${fn(n)}(cg: i32, y: i32, x: i32) -> vec4<i32> { return ${ev(n)}(cg, y, x); }`);
    }
  };

  const post = (n, acc) => {
    let v = sat(rs(acc, n.sh));
    if (n.act) v = relu(v);
    return v;
  };

  const emitEval = n => {
    let body;
    if (n.op === 'conv') {
      const X = n.ins[0], [kh, kw] = n.k, cog = n.CG, icg = X.CG;
      const inner = (iy, ix, wb) => `    for (var ig = 0; ig < ${icg}; ig++) {
      let v = ${fn(X)}(ig, ${iy}, ${ix});
      let w = ${wb} + ig * ${4 * cog};
      acc += v.x * wts[w] + v.y * wts[w + ${cog}] + v.z * wts[w + ${2 * cog}] + v.w * wts[w + ${3 * cog}];
    }`;
      if (kh === 1 && kw === 1) {
        body = `  var acc = wts[${wB + n.bOff} + cg];
  let iy = y * ${n.s} - ${n.p}; let ix = x * ${n.s} - ${n.p};
  {
${inner('iy', 'ix', `${wB + n.wOff} + cg`)}
  }`;
      } else {
        body = `  var acc = wts[${wB + n.bOff} + cg];
  for (var ky = 0; ky < ${kh}; ky++) {
    let iy = y * ${n.s} - ${n.p} + ky;
    for (var kx = 0; kx < ${kw}; kx++) {
    let ix = x * ${n.s} - ${n.p} + kx;
${inner('iy', 'ix', `${wB + n.wOff} + (ky * ${kw} + kx) * ${4 * icg * cog} + cg`)}
    }
  }`;
      }
      body += `\n  return ${post(n, 'acc')};`;
    } else if (n.op === 'dw') {
      const X = n.ins[0], [kh, kw] = n.k;
      body = `  var acc = wts[${wB + n.bOff} + cg];
  for (var ky = 0; ky < ${kh}; ky++) {
    let iy = y * ${n.s} - ${n.p} + ky;
    for (var kx = 0; kx < ${kw}; kx++) {
      acc += ${fn(X)}(cg, iy, x * ${n.s} - ${n.p} + kx) * wts[${wB + n.wOff} + (ky * ${kw} + kx) * ${n.CG} + cg];
    }
  }
  return ${post(n, 'acc')};`;
    } else if (n.op === 'add') {
      const [A, B] = n.ins;
      let v = sat(`${rs(`${fn(A)}(cg, y, x)`, n.sa)} + ${rs(`${fn(B)}(cg, y, x)`, n.sb)}`);
      if (n.act) v = relu(v);
      body = `  return ${v};`;
    } else if (n.op === 'concat') {
      let c0 = 0;
      const parts = n.ins.map((X, i) => {
        const s = n.shifts[i], e = s === 0 ? `${fn(X)}(cg - ${c0}, y, x)` : sat(rs(`${fn(X)}(cg - ${c0}, y, x)`, s));
        const line = i === n.ins.length - 1 ? `  return ${e};` : `  if (cg < ${c0 + X.CG}) { return ${e}; }`;
        c0 += X.CG;
        return line;
      });
      body = parts.join('\n');
    } else throw new Error(`no eval for ${n.op}`);
    out.push(`fn ${ev(n)}(cg: i32, y: i32, x: i32) -> vec4<i32> {\n${body}\n}`);
  };

  for (const x of F.ins) emitRead(x);
  emitEval(F);

  // Kernel body: stage shared nodes (zero outside the image), then the output tile.
  const tx = Math.ceil(F.W / T), ty = Math.ceil(F.H / T), tiles = tx * ty, cgPer = F.CG / S, wg = ctx.wg;
  const lines = [];
  lines.push(`fn ${p}main(wgi: i32, lid: i32) {
  let z = wgi / ${tiles}; let t = wgi % ${tiles};
  b = z / ${S}; let split = z % ${S};
  ty0 = (t / ${tx}) * ${T}; tx0 = (t % ${tx}) * ${T};`);
  if (F.toGap) lines.push(`  for (var i = lid; i < ${cgPer * 4}; i += ${wg}) { atomicStore(&gs[i], 0); }`);
  for (const n of plan.stages) {
    const r = R.get(n), lh = r.y1 - r.y0, lw = r.x1 - r.x0, N = n.CG * lh * lw;
    lines.push(`  for (var u = lid; u < ${N}; u += ${wg}) {
    let xx = u % ${lw}; let q = u / ${lw}; let yy = q % ${lh}; let cg = q / ${lh};
    let ay = ty0 * ${r.scale} + ${r.y0} + yy; let ax = tx0 * ${r.scale} + ${r.x0} + xx;
    var v = vec4<i32>(0);
    if (ay >= 0 && ay < ${n.H} && ax >= 0 && ax < ${n.W}) { v = ${ev(n)}(cg, ay, ax); }
    sh[${k.shOff.get(n)} + u] = v;
  }
  workgroupBarrier();`);
  }
  const store = F.toGap
    ? `      let o = (cg - split * ${cgPer}) * 4;
      atomicAdd(&gs[o], v.x); atomicAdd(&gs[o + 1], v.y); atomicAdd(&gs[o + 2], v.z); atomicAdd(&gs[o + 3], v.w);`
    : `      act[${F.aOff} + ((b * ${F.CG} + cg) * ${F.H} + ay) * ${F.W} + ax] = v;`;
  lines.push(`  for (var u = lid; u < ${cgPer * T * T}; u += ${wg}) {
    let xx = u % ${T}; let q = u / ${T}; let yy = q % ${T}; let cg = split * ${cgPer} + q / ${T};
    let ay = ty0 + yy; let ax = tx0 + xx;
    if (ay < ${F.H} && ax < ${F.W}) {
      let v = ${ev(F)}(cg, ay, ax);
${store}
    }
  }`);
  if (F.toGap) lines.push(`  workgroupBarrier();
  for (var i = lid; i < ${cgPer * 4}; i += ${wg}) { atomicAdd(&sums[${F.sumOff} + b * ${F.C} + split * ${cgPer * 4} + i], atomicLoad(&gs[i])); }`);
  lines.push('}');
  return out.join('\n') + '\n' + lines.join('\n');
}

// One module per dispatch: kernels picked by workgroup index; every kernel's grid scales with the batch.
export function moduleSource(kernels, ctx) {
  let shVec = 1, gsLen = 1;
  for (const k of kernels) {
    let off = 0;
    k.shOff = new Map();
    for (const n of k.group.plan.stages) {
      const r = k.group.plan.R.get(n);
      k.shOff.set(n, off);
      off += n.CG * (r.y1 - r.y0) * (r.x1 - r.x0);
    }
    shVec = Math.max(shVec, off);
    if (k.group.F.toGap) gsLen = Math.max(gsLen, k.group.F.C / k.group.plan.S);
  }
  const head = [`struct Cfg { batch: i32, pad0: i32, pad1: i32, pad2: i32 }
@group(0) @binding(0) var<storage, read> wts: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read_write> act: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read_write> sums: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> cfg: Cfg;
var<workgroup> sh: array<vec4<i32>, ${shVec}>;
var<workgroup> gs: array<atomic<i32>, ${gsLen}>;
var<private> b: i32;
var<private> ty0: i32;
var<private> tx0: i32;`];
  if (ctx.source === 'texture') head.push(CROP);
  const bodies = kernels.map((k, j) => kernelSource(k, j, ctx));
  let acc = 0;
  const dispatch = kernels.map((k, j) => {
    acc += k.group.plan.nwg;
    return `  if (w < cfg.batch * ${acc}) { k${j}_main(w - cfg.batch * ${acc - k.group.plan.nwg}, lid); return; }`;
  });
  const entry = `@compute @workgroup_size(${ctx.wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let w = i32(wid.x); let lid = i32(li);
${dispatch.join('\n')}
}`;
  return { code: [...head, ...bodies, entry].join('\n\n'), wgPerItem: acc, sharedBytes: shVec * 16 + gsLen * 4 };
}

// Bilinear resample of the video frame, matching a canvas drawImage with an affine transform composited on
// black: texels outside the frame are transparent black. Output is BGR - 128, like the reference tensor.
const CROP = `struct Crop { m: vec4<f32>, t: vec2<f32>, lvl: i32, pad: i32 }
@group(0) @binding(4) var frame: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> crops: array<Crop>;
fn texel(p: vec2<i32>, lvl: i32, size: vec2<i32>) -> vec3<f32> {
  if (p.x < 0 || p.y < 0 || p.x >= size.x || p.y >= size.y) { return vec3<f32>(0.0); }
  return textureLoad(frame, p, lvl).rgb;
}
fn crop_px(i: i32, y: i32, x: i32) -> vec4<i32> {
  let c = crops[i];
  let d = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
  let s = vec2<f32>(c.m.x * d.x + c.m.z * d.y + c.t.x, c.m.y * d.x + c.m.w * d.y + c.t.y) - vec2<f32>(0.5);
  let f = floor(s); let fr = s - f; let p = vec2<i32>(f);
  let size = vec2<i32>(textureDimensions(frame, c.lvl));
  let top = mix(texel(p, c.lvl, size), texel(p + vec2<i32>(1, 0), c.lvl, size), fr.x);
  let bot = mix(texel(p + vec2<i32>(0, 1), c.lvl, size), texel(p + vec2<i32>(1, 1), c.lvl, size), fr.x);
  let rgb = vec3<i32>(round(clamp(mix(top, bot, fr.y), vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0));
  return vec4<i32>(rgb.b - 128, rgb.g - 128, rgb.r - 128, 0);
}`;

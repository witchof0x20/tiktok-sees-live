// WGSL generator. Every tensor is NC4HW4 (vec4<i32> per 4 channels per pixel).
//
// Each kernel computes one output tile. Nodes are emitted as functions of two kinds:
//  - element reads  e(cg, y, x) -> vec4          global (zero outside the image = padding), workgroup memory,
//                                                or a pointwise op recomputed from its inputs
//  - block evals    b(cg0, y, x0) -> P*Q vec4     a register tile of Q channel groups x P adjacent pixels;
//                                                convs load each input once per tile and each weight once
//                                                per P pixels, depthwise rows slide over shared loads.
// Nodes staged in workgroup memory are computed block-wise in their own phase; the kernel output (and any
// pointwise chain feeding it) is computed block-wise straight into registers. Several kernels can share a
// module and be selected by workgroup index (horizontal fusion).

function rs(e, s) {
  if (s === 0) return e;
  if (s > 0) return `((${e} + vec4<i32>(${1 << (s - 1)})) >> vec4<u32>(${s}u))`;
  return `(${e} * vec4<i32>(${1 << -s}))`;
}
const sat = e => `clamp(${e}, vec4<i32>(-32768), vec4<i32>(32767))`;
const relu = e => `max(${e}, vec4<i32>(0))`;
const post = (n, e) => { let v = sat(rs(e, n.sh)); if (n.act) v = relu(v); return v; };
const range = n => Array.from({ length: n }, (_, i) => i);

function kernelSource(k, j, ctx) {
  const { members, F, plan } = k.group, { mode, R, T, S, P, Q } = plan;
  const p = `k${j}_`, wB = k.wBase;
  const out = [], done = new Set();
  const once = (name, f) => { if (!done.has(name)) { done.add(name); f(); } return name; };

  // Element read of n at absolute (cg, y, x).
  const elem = n => once(`${p}e${n.idx}`, () => {
    const name = `${p}e${n.idx}`;
    if (!members.has(n)) {
      const body = n.op === 'data' && ctx.source === 'texture'
        ? `return crop_px(b * ${ctx.nets} + ${k.net}, y, x);`
        : `return act[${n.aOff} + ((b * ${n.CG} + cg) * ${n.H} + y) * ${n.W} + x];`;
      out.push(`fn ${name}(cg: i32, y: i32, x: i32) -> vec4<i32> {
  if (y < 0 || y >= ${n.H} || x < 0 || x >= ${n.W}) { return vec4<i32>(0); }
  ${body}
}`);
    } else if (mode.get(n) === 'shared') {
      const r = R.get(n), lh = r.y1 - r.y0, lw = r.x1 - r.x0;
      out.push(`fn ${name}(cg: i32, y: i32, x: i32) -> vec4<i32> {
  return sh[${k.shOff.get(n)} + (cg * ${lh} + (y - (ty0 * ${r.scale} + ${r.y0}))) * ${lw} + (x - (tx0 * ${r.scale} + ${r.x0}))];
}`);
    } else if (n.op === 'add') {
      const [A, B] = n.ins.map(elem);
      let v = sat(`${rs(`${A}(cg, y, x)`, n.sa)} + ${rs(`${B}(cg, y, x)`, n.sb)}`);
      if (n.act) v = relu(v);
      out.push(`fn ${name}(cg: i32, y: i32, x: i32) -> vec4<i32> { return ${v}; }`);
    } else if (n.op === 'concat') {
      let c0 = 0;
      const lines = n.ins.map((X, i) => {
        const s = n.shifts[i], call = `${elem(X)}(cg - ${c0}, y, x)`, e = s === 0 ? call : sat(rs(call, s));
        const l = i === n.ins.length - 1 ? `  return ${e};` : `  if (cg < ${c0 + X.CG}) { return ${e}; }`;
        c0 += X.CG;
        return l;
      });
      out.push(`fn ${name}(cg: i32, y: i32, x: i32) -> vec4<i32> {\n${lines.join('\n')}\n}`);
    } else {
      const bl = block(n, 1, 1);
      out.push(`fn ${name}(cg: i32, y: i32, x: i32) -> vec4<i32> { return ${bl}(cg, y, x)[0]; }`);
    }
  });

  // Block eval of n: Q channel groups from cg0, P pixels from x0, row y. Result index q * P + i.
  // compute=true evaluates a staged node from its inputs (its own stage); otherwise staged nodes are read back.
  const block = (n, BP, BQ, compute = false) => once(`${p}b${n.idx}_${BP}_${BQ}${compute ? 'c' : ''}`, () => {
    const name = `${p}b${n.idx}_${BP}_${BQ}${compute ? 'c' : ''}`, N = BP * BQ, L = [];
    const ret = `array<vec4<i32>, ${N}>`;
    const at = (q, i) => q * BP + i;
    if (!members.has(n) || (mode.get(n) === 'shared' && !compute)) {
      const e = elem(n);
      L.push(`  return ${ret}(${range(BQ).flatMap(q => range(BP).map(i => `${e}(cg0 + ${q}, y, x0 + ${i})`)).join(', ')});`);
    } else if (n.op === 'add') {
      const A = block(n.ins[0], BP, BQ), B = block(n.ins[1], BP, BQ);
      L.push(`  let a = ${A}(cg0, y, x0); let c = ${B}(cg0, y, x0);`);
      const el = t => { let v = sat(`${rs(`a[${t}]`, n.sa)} + ${rs(`c[${t}]`, n.sb)}`); return n.act ? relu(v) : v; };
      L.push(`  return ${ret}(${range(N).map(el).join(', ')});`);
    } else if (n.op === 'concat') {
      let c0 = 0;
      n.ins.forEach((X, i) => {
        const s = n.shifts[i], bx = block(X, BP, BQ);
        const val = s === 0 ? `${bx}(cg0 - ${c0}, y, x0)` : null;
        const body = val ? `return ${val};` : `let a = ${bx}(cg0 - ${c0}, y, x0); return ${ret}(${range(N).map(t => sat(rs(`a[${t}]`, s))).join(', ')});`;
        L.push(i === n.ins.length - 1 ? `  ${body}` : `  if (cg0 < ${c0 + X.CG}) { ${body} }`);
        c0 += X.CG;
      });
    } else if (n.op === 'conv' && n.u24) {
      // Unsigned-offset operands keep every multiply within 24 bits; all sums wrap mod 2^32 and the true
      // accumulator (|acc| < 2^30) comes back exactly after removing 2048 * sum(x').
      const X = n.ins[0], ex = elem(X), [kh, kw] = n.k, cog = n.CG, icg = X.CG, M16 = 'vec4<u32>(0xffffu)', M12 = 'vec4<u32>(0xfffu)';
      L.push(...range(BQ).map(q => `  let bias${q} = bitcast<vec4<u32>>(wts[${wB + n.bOff} + cg0 + ${q}]);`));
      L.push(...range(N).map(t => `  var a${t} = bias${(t / BP) | 0};`));
      L.push(...range(BP).map(i => `  var s${i} = 0u;`));
      L.push(`  for (var ky = 0; ky < ${kh}; ky++) {
  let iy = y * ${n.s} - ${n.p} + ky;
  for (var kx = 0; kx < ${kw}; kx++) {
  let ix = x0 * ${n.s} - ${n.p} + kx;
  for (var ig = 0; ig < ${icg}; ig++) {
    let wb = ${wB + n.wOff} + (ky * ${kw} + kx) * ${4 * icg * cog} + ig * ${4 * cog} + cg0;`);
      for (const i of range(BP)) {
        L.push(`    let v${i} = bitcast<vec4<u32>>(${ex}(ig, iy, ix + ${i * n.s}) + vec4<i32>(32768)) & ${M16};`);
        L.push(`    s${i} += v${i}.x + v${i}.y + v${i}.z + v${i}.w;`);
      }
      for (const q of range(BQ)) {
        L.push(`    { let w0 = bitcast<vec4<u32>>(wts[wb + ${q}]) & ${M12}; let w1 = bitcast<vec4<u32>>(wts[wb + ${cog + q}]) & ${M12}; let w2 = bitcast<vec4<u32>>(wts[wb + ${2 * cog + q}]) & ${M12}; let w3 = bitcast<vec4<u32>>(wts[wb + ${3 * cog + q}]) & ${M12};`);
        L.push(...range(BP).map(i => `      a${at(q, i)} += v${i}.x * w0 + v${i}.y * w1 + v${i}.z * w2 + v${i}.w * w3;`));
        L.push('    }');
      }
      L.push('  }\n  }\n  }');
      L.push(`  return ${ret}(${range(N).map(t => post(n, `bitcast<vec4<i32>>(a${t} - vec4<u32>(s${t % BP} * 2048u))`)).join(', ')});`);
    } else if (n.op === 'conv') {
      const X = n.ins[0], ex = elem(X), [kh, kw] = n.k, cog = n.CG, icg = X.CG;
      L.push(...range(BQ).map(q => `  let bias${q} = wts[${wB + n.bOff} + cg0 + ${q}];`));
      L.push(...range(N).map(t => `  var a${t} = bias${(t / BP) | 0};`));
      L.push(`  for (var ky = 0; ky < ${kh}; ky++) {
  let iy = y * ${n.s} - ${n.p} + ky;
  for (var kx = 0; kx < ${kw}; kx++) {
  let ix = x0 * ${n.s} - ${n.p} + kx;
  for (var ig = 0; ig < ${icg}; ig++) {
    let wb = ${wB + n.wOff} + (ky * ${kw} + kx) * ${4 * icg * cog} + ig * ${4 * cog} + cg0;`);
      L.push(...range(BP).map(i => `    let v${i} = ${ex}(ig, iy, ix + ${i * n.s});`));
      for (const q of range(BQ)) {
        L.push(`    { let w0 = wts[wb + ${q}]; let w1 = wts[wb + ${cog + q}]; let w2 = wts[wb + ${2 * cog + q}]; let w3 = wts[wb + ${3 * cog + q}];`);
        L.push(...range(BP).map(i => `      a${at(q, i)} += v${i}.x * w0 + v${i}.y * w1 + v${i}.z * w2 + v${i}.w * w3;`));
        L.push('    }');
      }
      L.push('  }\n  }\n  }');
      L.push(`  return ${ret}(${range(N).map(t => post(n, `a${t}`)).join(', ')});`);
    } else if (n.op === 'dw') {
      const X = n.ins[0], ex = elem(X), [kh, kw] = n.k, span = (BP - 1) * n.s + kw;
      L.push(...range(N).map(t => `  var a${t} = wts[${wB + n.bOff} + cg0 + ${(t / BP) | 0}];`));
      L.push(`  for (var ky = 0; ky < ${kh}; ky++) {
  let iy = y * ${n.s} - ${n.p} + ky;
  let ix = x0 * ${n.s} - ${n.p};`);
      for (const q of range(BQ)) {
        // One row of the receptive field, loaded once and reused by every tap and pixel.
        L.push(`  {`);
        L.push(...range(span).map(t => `    let r${t} = ${ex}(cg0 + ${q}, iy, ix + ${t});`));
        for (const kx of range(kw)) {
          L.push(`    { let w = wts[${wB + n.wOff} + (ky * ${kw} + ${kx}) * ${n.CG} + cg0 + ${q}];`);
          L.push(...range(BP).map(i => `      a${at(q, i)} += r${i * n.s + kx} * w;`));
          L.push('    }');
        }
        L.push('  }');
      }
      L.push('  }');
      L.push(`  return ${ret}(${range(N).map(t => post(n, `a${t}`)).join(', ')});`);
    } else throw new Error(`no block eval for ${n.op}`);
    out.push(`fn ${name}(cg0: i32, y: i32, x0: i32) -> ${ret} {\n${L.join('\n')}\n}`);
  });

  const fb = block(F, P, Q);
  const stageFns = plan.stages.map(s => block(s.n, s.P, s.Q, true));

  const tx = Math.ceil(F.W / T), ty = Math.ceil(F.H / T), tiles = tx * ty, cgPer = F.CG / S, wg = ctx.wg;
  const lines = [`fn ${p}main(wgi: i32, lid: i32) {
  let z = wgi / ${tiles}; let t = wgi % ${tiles};
  b = z / ${S}; let split = z % ${S};
  ty0 = (t / ${tx}) * ${T}; tx0 = (t % ${tx}) * ${T};`];
  if (F.toGap) lines.push(`  for (var i = lid; i < ${cgPer * 4}; i += ${wg}) { atomicStore(&gs[i], 0); }`);
  const px = plan.sched === 'px';
  plan.stages.forEach((s, si) => {
    const n = s.n, r = R.get(n), lh = r.y1 - r.y0, lw = r.x1 - r.x0, xb = Math.ceil(lw / s.P), N = (n.CG / s.Q) * lh * xb;
    lines.push(px ? `  for (var cb = 0; cb < ${n.CG / s.Q}; cb++) {
  for (var u = lid; u < ${lh * xb}; u += ${wg}) {
    let xi = u % ${xb}; let yy = u / ${xb}; let cg0 = cb * ${s.Q};` : `  {
  for (var u = lid; u < ${N}; u += ${wg}) {
    let xi = u % ${xb}; let q = u / ${xb}; let yy = q % ${lh}; let cg0 = (q / ${lh}) * ${s.Q};`);
    lines.push(`    let ay = ty0 * ${r.scale} + ${r.y0} + yy; let ax = tx0 * ${r.scale} + ${r.x0} + xi * ${s.P};
    let v = ${stageFns[si]}(cg0, ay, ax);
    let inY = ay >= 0 && ay < ${n.H};`);
    for (const q of range(s.Q)) for (const i of range(s.P)) {
      const guard = s.P * xb > lw ? `xi * ${s.P} + ${i} < ${lw}` : 'true';
      lines.push(`    if (${guard}) { sh[${k.shOff.get(n)} + ((cg0 + ${q}) * ${lh} + yy) * ${lw} + xi * ${s.P} + ${i}] = select(vec4<i32>(0), v[${q * s.P + i}], inY && ax + ${i} >= 0 && ax + ${i} < ${n.W}); }`);
    }
    lines.push(`  }
  }
  workgroupBarrier();`);
  });
  const xbF = Math.ceil(T / P), NF = (cgPer / Q) * T * xbF;
  lines.push(px ? `  for (var cb = 0; cb < ${cgPer / Q}; cb++) {
  for (var u = lid; u < ${T * xbF}; u += ${wg}) {
    let xi = u % ${xbF}; let yy = u / ${xbF}; let cg0 = split * ${cgPer} + cb * ${Q};` : `  {
  for (var u = lid; u < ${NF}; u += ${wg}) {
    let xi = u % ${xbF}; let q = u / ${xbF}; let yy = q % ${T}; let cg0 = split * ${cgPer} + (q / ${T}) * ${Q};`);
  lines.push(`    let ay = ty0 + yy; let ax = tx0 + xi * ${P};
    if (ay < ${F.H}) {
    let v = ${fb}(cg0, ay, ax);`);
  for (const q of range(Q)) for (const i of range(P)) {
    const guard = `ax + ${i} < ${F.W}` + (P * xbF > T ? ` && xi * ${P} + ${i} < ${T}` : '');
    if (F.toGap) lines.push(`    if (${guard}) { let o = (cg0 + ${q} - split * ${cgPer}) * 4; let w = v[${q * P + i}]; atomicAdd(&gs[o], w.x); atomicAdd(&gs[o + 1], w.y); atomicAdd(&gs[o + 2], w.z); atomicAdd(&gs[o + 3], w.w); }`);
    else lines.push(`    if (${guard}) { act[${F.aOff} + ((b * ${F.CG} + cg0 + ${q}) * ${F.H} + ay) * ${F.W} + ax + ${i}] = v[${q * P + i}]; }`);
  }
  lines.push('    }\n  }\n  }');
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
    for (const s of k.group.plan.stages) {
      const r = k.group.plan.R.get(s.n);
      k.shOff.set(s.n, off);
      off += s.n.CG * (r.y1 - r.y0) * (r.x1 - r.x0);
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

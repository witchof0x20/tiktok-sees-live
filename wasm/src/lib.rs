//! CPU inference for TikTok's decoded face nets (the fallback when WebGPU isn't available), compiled to
//! WebAssembly with SIMD128.
//!
//! Semantics are the nets' 16-bit fixed point, exactly as the reference engine computes it: int16 activations,
//! int32 accumulation, round-half-up rescale, int16 saturation, ReLU. Every accumulator provably stays below 2^30
//! (tools/bounds.mjs), so i32 arithmetic is exact and results are bit-identical to the reference.
//!
//! Tensors are stored channel-pair interleaved: element (c, p) at `(c / 2) * hwp * 2 + p * 2 + (c % 2)`, with the
//! pixel count padded to `hwp` (a multiple of 8, plus 8 lanes of slack for vector stores at row ends). A SIMD load then holds 4 pixels x 2 channels, which is exactly
//! the operand shape of `i32x4.dot_i16x8_s`, so a 1x1 conv does 8 multiply-adds per instruction.
//!
//! The program (shapes, ops, weights) comes from JS (live/cpu/program.js). It is fully validated in `Prog::parse`
//! before anything runs, so the kernels can index without further checks.

const MAGIC: u32 = 0x544b_4e31; // "TKN1"

#[derive(Clone, Copy, Debug)]
struct Shape {
    c: usize,
    h: usize,
    w: usize,
    hwp: usize,
}

impl Shape {
    fn len(&self) -> usize {
        self.c.div_ceil(2) * self.hwp * 2
    }
}

#[derive(Debug)]
enum Op {
    /// 1x1 conv, stride s, no padding. Weights [oc][ic_pad] (ic padded to even), bias pre-shifted.
    Pw { out: usize, inp: usize, s: usize, sh: u32, act: bool, w: usize, b: usize },
    /// General kxk conv. Weights [oc][tap][ic] in the program; kept here as [oc][kp] with kp = taps*ic padded to 8.
    Conv { out: usize, inp: usize, kh: usize, kw: usize, s: usize, p: usize, sh: u32, act: bool, b: usize, wk: Vec<i16>, kp: usize },
    /// Depthwise kxk. Weights [c][tap].
    Dw { out: usize, inp: usize, kh: usize, kw: usize, s: usize, p: usize, sh: u32, act: bool, w: usize, b: usize },
    Add { out: usize, a: usize, b: usize, sa: u32, sb: u32, act: bool },
    Concat { out: usize, parts: Vec<(usize, u32)> },
    Gap { inp: usize, off: usize },
}

pub struct Prog {
    shapes: Vec<Shape>,
    ops: Vec<Op>,
    w: Vec<i16>,
    b: Vec<i32>,
    t: Vec<Vec<i16>>,
    scratch: Vec<i16>,
    pad: Vec<i16>,
    input_t: usize,
    input: Vec<i16>,
    sums: Vec<i32>,
}

struct Words<'a> {
    w: &'a [u32],
    i: usize,
}
impl Words<'_> {
    fn next(&mut self) -> Option<u32> {
        let v = *self.w.get(self.i)?;
        self.i += 1;
        Some(v)
    }
    fn usize(&mut self) -> Option<usize> {
        self.next().map(|v| v as usize)
    }
}

fn out_dim(n: usize, k: usize, s: usize, p: usize) -> Option<usize> {
    if s == 0 || n + 2 * p < k {
        return None;
    }
    Some((n + 2 * p - k) / s + 1)
}

impl Prog {
    /// Validates and builds a program. Returns None on any inconsistency.
    pub fn parse(words: &[u32], w: Vec<i16>, b: Vec<i32>) -> Option<Prog> {
        let mut r = Words { w: words, i: 0 };
        if r.next()? != MAGIC || r.next()? != 1 {
            return None;
        }
        let (nt, nops, input_t, n_sums) = (r.usize()?, r.usize()?, r.usize()?, r.usize()?);
        if nt == 0 || nt > 4096 || nops > 4096 || n_sums > 1 << 16 || input_t >= nt {
            return None;
        }
        let mut shapes = Vec::with_capacity(nt);
        for _ in 0..nt {
            let (c, h, w) = (r.usize()?, r.usize()?, r.usize()?);
            if c == 0 || c % 2 != 0 || h == 0 || w == 0 || c > 4096 || h > 4096 || w > 4096 || h * w > 1 << 20 {
                return None;
            }
            shapes.push(Shape { c, h, w, hwp: (h * w).div_ceil(8) * 8 + 8 });
        }
        let tid = |v: usize| if v < nt { Some(v) } else { None };
        let shift = |v: u32| if v <= 24 { Some(v) } else { None };
        let mut ops = Vec::with_capacity(nops);
        let mut written = vec![false; nt];
        written[input_t] = true;
        let mut sums_used = vec![false; n_sums];
        for _ in 0..nops {
            let kind = r.next()?;
            let op = match kind {
                1 => {
                    let (out, inp, s, sh, act, wo, bo) = (tid(r.usize()?)?, tid(r.usize()?)?, r.usize()?, shift(r.next()?)?, r.next()? != 0, r.usize()?, r.usize()?);
                    let (i, o) = (shapes[inp], shapes[out]);
                    if out_dim(i.h, 1, s, 0)? != o.h || out_dim(i.w, 1, s, 0)? != o.w || o.c % 4 != 0 || sh == 0 {
                        return None;
                    }
                    if wo.checked_add(o.c * i.c)? > w.len() || wo % 2 != 0 || bo.checked_add(o.c)? > b.len() {
                        return None;
                    }
                    Op::Pw { out, inp, s, sh, act, w: wo, b: bo }
                }
                2 | 3 => {
                    let (out, inp, kh, kw, s, p, sh, act, wo, bo) = (tid(r.usize()?)?, tid(r.usize()?)?, r.usize()?, r.usize()?, r.usize()?, r.usize()?, shift(r.next()?)?, r.next()? != 0, r.usize()?, r.usize()?);
                    let (i, o) = (shapes[inp], shapes[out]);
                    if kh == 0 || kw == 0 || kh > 16 || kw > 16 || p > 16 || out_dim(i.h, kh, s, p)? != o.h || out_dim(i.w, kw, s, p)? != o.w {
                        return None;
                    }
                    let wn = if kind == 2 { o.c * kh * kw * i.c } else { o.c * kh * kw };
                    if (kind == 3 && i.c != o.c) || wo.checked_add(wn)? > w.len() || bo.checked_add(o.c)? > b.len() {
                        return None;
                    }
                    if kind == 2 {
                        let k = kh * kw * i.c;
                        let kp = k.div_ceil(8) * 8;
                        let mut wk = vec![0i16; o.c * kp];
                        for oc in 0..o.c {
                            wk[oc * kp..oc * kp + k].copy_from_slice(&w[wo + oc * k..wo + (oc + 1) * k]);
                        }
                        Op::Conv { out, inp, kh, kw, s, p, sh, act, b: bo, wk, kp }
                    } else {
                        Op::Dw { out, inp, kh, kw, s, p, sh, act, w: wo, b: bo }
                    }
                }
                4 => {
                    let (out, a, bb, sa, sb, act) = (tid(r.usize()?)?, tid(r.usize()?)?, tid(r.usize()?)?, shift(r.next()?)?, shift(r.next()?)?, r.next()? != 0);
                    let (x, y, o) = (shapes[a], shapes[bb], shapes[out]);
                    if x.c != o.c || y.c != o.c || x.h != o.h || y.h != o.h || x.w != o.w || y.w != o.w || !written[a] || !written[bb] {
                        return None;
                    }
                    Op::Add { out, a, b: bb, sa, sb, act }
                }
                5 => {
                    let out = tid(r.usize()?)?;
                    let n = r.usize()?;
                    if n == 0 || n > 16 {
                        return None;
                    }
                    let mut parts = Vec::with_capacity(n);
                    let mut c = 0;
                    for _ in 0..n {
                        let (t, s) = (tid(r.usize()?)?, shift(r.next()?)?);
                        let x = shapes[t];
                        if x.h != shapes[out].h || x.w != shapes[out].w || !written[t] {
                            return None;
                        }
                        c += x.c;
                        parts.push((t, s));
                    }
                    if c != shapes[out].c {
                        return None;
                    }
                    Op::Concat { out, parts }
                }
                6 => {
                    let (inp, off) = (tid(r.usize()?)?, r.usize()?);
                    if !written[inp] || off.checked_add(shapes[inp].c)? > n_sums || sums_used[off..off + shapes[inp].c].iter().any(|&u| u) {
                        return None;
                    }
                    sums_used[off..off + shapes[inp].c].iter_mut().for_each(|u| *u = true);
                    Op::Gap { inp, off }
                }
                _ => return None,
            };
            // Inputs must be produced before use; an op's output must not alias one of its inputs.
            let (out, ins): (Option<usize>, Vec<usize>) = match &op {
                Op::Pw { out, inp, .. } | Op::Conv { out, inp, .. } | Op::Dw { out, inp, .. } => (Some(*out), vec![*inp]),
                Op::Add { out, a, b, .. } => (Some(*out), vec![*a, *b]),
                Op::Concat { out, parts } => (Some(*out), parts.iter().map(|p| p.0).collect()),
                Op::Gap { inp, .. } => (None, vec![*inp]),
            };
            if ins.iter().any(|&i| !written[i] || Some(i) == out) {
                return None;
            }
            if let Some(o) = out {
                written[o] = true;
            }
            ops.push(op);
        }
        if r.i != words.len() {
            return None;
        }
        let t: Vec<Vec<i16>> = shapes.iter().map(|s| vec![0i16; s.len()]).collect();
        let scratch_len = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let ins = shapes[input_t];
        Some(Prog { input: vec![0; ins.h * ins.w * (ins.c - 1).max(1)], shapes, ops, w, b, t, scratch: vec![0; scratch_len], pad: Vec::new(), input_t, sums: vec![0; n_sums] })
    }

    /// HWC input with (c - 1) channels (the input tensor is padded to an even count; 3 -> 4).
    pub fn input_mut(&mut self) -> &mut [i16] {
        &mut self.input
    }
    pub fn sums(&self) -> &[i32] {
        &self.sums
    }

    pub fn run(&mut self) {
        let s = self.shapes[self.input_t];
        let cin = s.c - 1;
        let t = &mut self.t[self.input_t];
        t.fill(0);
        for p in 0..s.h * s.w {
            for c in 0..cin {
                t[(c >> 1) * s.hwp * 2 + p * 2 + (c & 1)] = self.input[p * cin + c];
            }
        }
        self.sums.fill(0);
        for k in 0..self.ops.len() {
            match self.ops[k] {
                Op::Pw { out, inp, s, sh, act, w, b } => {
                    let (si, so) = (self.shapes[inp], self.shapes[out]);
                    let x = core::mem::take(&mut self.t[inp]);
                    let mut y = core::mem::take(&mut self.t[out]);
                    if s == 1 {
                        pw(&x, si, &mut y, so, &self.w[w..w + so.c * si.c], &self.b[b..b + so.c], sh, act);
                    } else {
                        // Strided 1x1: gather the sampled pixels into scratch first, then the same kernel.
                        let sub = Shape { c: si.c, h: so.h, w: so.w, hwp: so.hwp };
                        let tmp = &mut self.scratch[..sub.len()];
                        tmp.fill(0);
                        for cp in 0..si.c / 2 {
                            for oy in 0..so.h {
                                for ox in 0..so.w {
                                    let ip = (oy * s) * si.w + ox * s;
                                    let (d, src) = (cp * sub.hwp * 2 + (oy * so.w + ox) * 2, cp * si.hwp * 2 + ip * 2);
                                    tmp[d] = x[src];
                                    tmp[d + 1] = x[src + 1];
                                }
                            }
                        }
                        pw(tmp, sub, &mut y, so, &self.w[w..w + so.c * si.c], &self.b[b..b + so.c], sh, act);
                    }
                    self.t[inp] = x;
                    self.t[out] = y;
                }
                Op::Conv { out, inp, kh, kw, s, p, sh, act, b, ref wk, kp } => {
                    let (si, so) = (self.shapes[inp], self.shapes[out]);
                    let x = core::mem::take(&mut self.t[inp]);
                    let mut y = core::mem::take(&mut self.t[out]);
                    conv_im2col(&x, si, &mut y, so, wk, kp, &self.b[b..b + so.c], kh, kw, s, p, sh, act);
                    self.t[inp] = x;
                    self.t[out] = y;
                }
                Op::Dw { out, inp, kh, kw, s, p, sh, act, w, b } => {
                    let (si, so) = (self.shapes[inp], self.shapes[out]);
                    let x = core::mem::take(&mut self.t[inp]);
                    let mut y = core::mem::take(&mut self.t[out]);
                    dw_padded(&x, si, &mut y, so, &self.w[w..w + so.c * kh * kw], &self.b[b..b + so.c], kh, kw, s, p, sh, act, &mut self.pad);
                    self.t[inp] = x;
                    self.t[out] = y;
                }
                Op::Add { out, a, b, sa, sb, act } => {
                    let n = self.shapes[out].len();
                    let mut y = core::mem::take(&mut self.t[out]);
                    {
                        let (x0, x1) = (&self.t[a], &self.t[b]);
                        for q in 0..n {
                            let v = sat(rs(x0[q] as i32, sa) + rs(x1[q] as i32, sb));
                            y[q] = if act && v < 0 { 0 } else { v as i16 };
                        }
                    }
                    self.t[out] = y;
                }
                Op::Concat { out, ref parts } => {
                    let mut y = core::mem::take(&mut self.t[out]);
                    let so = self.shapes[out];
                    let mut cp0 = 0;
                    for &(t, sft) in parts {
                        let si = self.shapes[t];
                        let x = &self.t[t];
                        for cp in 0..si.c / 2 {
                            for q in 0..so.hwp * 2 {
                                y[(cp0 + cp) * so.hwp * 2 + q] = sat(rs(x[cp * si.hwp * 2 + q] as i32, sft)) as i16;
                            }
                        }
                        cp0 += si.c / 2;
                    }
                    self.t[out] = y;
                }
                Op::Gap { inp, off } => {
                    let s = self.shapes[inp];
                    let x = &self.t[inp];
                    for c in 0..s.c {
                        let base = (c >> 1) * s.hwp * 2 + (c & 1);
                        let mut acc = 0i32;
                        for p in 0..s.h * s.w {
                            acc += x[base + p * 2] as i32;
                        }
                        self.sums[off + c] = acc;
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn rs(x: i32, s: u32) -> i32 {
    if s == 0 { x } else { (x + (1 << (s - 1))) >> s }
}
#[inline(always)]
fn sat(v: i32) -> i32 {
    v.clamp(-32768, 32767)
}

#[inline(always)]
fn at(sh: Shape, c: usize, pix: usize) -> usize {
    (c >> 1) * sh.hwp * 2 + pix * 2 + (c & 1)
}

/// General kxk conv as one im2col patch per output pixel (zeros outside the image) dotted with every output
/// channel's weights, 8 at a time.
#[allow(clippy::too_many_arguments)]
fn conv_im2col(x: &[i16], si: Shape, y: &mut [i16], so: Shape, wk: &[i16], kp: usize, b: &[i32], kh: usize, kw: usize, s: usize, p: usize, sh: u32, act: bool) {
    let mut patch = vec![0i16; kp];
    for oy in 0..so.h {
        for ox in 0..so.w {
            patch.fill(0);
            for ky in 0..kh {
                let iy = (oy * s + ky) as isize - p as isize;
                if iy < 0 || iy >= si.h as isize {
                    continue;
                }
                for kx in 0..kw {
                    let ix = (ox * s + kx) as isize - p as isize;
                    if ix < 0 || ix >= si.w as isize {
                        continue;
                    }
                    let pix = iy as usize * si.w + ix as usize;
                    let base = (ky * kw + kx) * si.c;
                    for ic in 0..si.c {
                        patch[base + ic] = x[at(si, ic, pix)];
                    }
                }
            }
            for oc in 0..so.c {
                let v = sat(rs(b[oc] + dot(&patch, &wk[oc * kp..(oc + 1) * kp]), sh));
                y[at(so, oc, oy * so.w + ox)] = if act && v < 0 { 0 } else { v as i16 };
            }
        }
    }
}

#[inline(always)]
fn dot(a: &[i16], b: &[i16]) -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        use core::arch::wasm32::*;
        let mut acc = i32x4_splat(0);
        for k in (0..a.len()).step_by(8) {
            // SAFETY: both slices have the same length, a multiple of 8.
            let (va, vb) = unsafe { (v128_load(a.as_ptr().add(k) as *const v128), v128_load(b.as_ptr().add(k) as *const v128)) };
            acc = i32x4_add(acc, i32x4_dot_i16x8(va, vb));
        }
        i32x4_extract_lane::<0>(acc) + i32x4_extract_lane::<1>(acc) + i32x4_extract_lane::<2>(acc) + i32x4_extract_lane::<3>(acc)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.iter().zip(b).map(|(&p, &q)| p as i32 * q as i32).sum()
    }
}

/// Depthwise kxk. Each channel pair is copied into a zero-bordered plane, so taps never need bounds checks; with
/// stride 1, one load covers 4 pixels x 2 channels and widening multiplies accumulate them in two i32x4.
#[allow(clippy::too_many_arguments)]
fn dw_padded(x: &[i16], si: Shape, y: &mut [i16], so: Shape, w: &[i16], b: &[i32], kh: usize, kw: usize, s: usize, p: usize, sh: u32, act: bool, pad: &mut Vec<i16>) {
    let taps = kh * kw;
    let (ph, pw_) = (si.h + 2 * p + 1, si.w + 2 * p + 8); // slack so 4-pixel loads at row ends stay inside
    pad.clear();
    pad.resize(ph * pw_ * 2, 0);
    for cp in 0..si.c / 2 {
        for iy in 0..si.h {
            let src = cp * si.hwp * 2 + iy * si.w * 2;
            let dst = ((iy + p) * pw_ + p) * 2;
            pad[dst..dst + si.w * 2].copy_from_slice(&x[src..src + si.w * 2]);
        }
        let (c0, c1) = (2 * cp, 2 * cp + 1);
        if s == 1 && dw_simd(pad, pw_, y, so, cp, &w[c0 * taps..(c1 + 1) * taps], [b[c0], b[c1]], kh, kw, sh, act) {
            continue;
        }
        for oy in 0..so.h {
            for ox in 0..so.w {
                let (mut a0, mut a1) = (b[c0], b[c1]);
                for ky in 0..kh {
                    let row = ((oy * s + ky) * pw_ + ox * s) * 2;
                    for kx in 0..kw {
                        a0 += w[c0 * taps + ky * kw + kx] as i32 * pad[row + kx * 2] as i32;
                        a1 += w[c1 * taps + ky * kw + kx] as i32 * pad[row + kx * 2 + 1] as i32;
                    }
                }
                let o = cp * so.hwp * 2 + (oy * so.w + ox) * 2;
                for (k, a) in [a0, a1].into_iter().enumerate() {
                    let v = sat(rs(a, sh));
                    y[o + k] = if act && v < 0 { 0 } else { v as i16 };
                }
            }
        }
    }
}

/// Stride-1 depthwise for one channel pair on its padded plane. Returns false where SIMD isn't available.
#[allow(clippy::too_many_arguments, unused_variables)]
fn dw_simd(pad: &[i16], pw_: usize, y: &mut [i16], so: Shape, cp: usize, w: &[i16], b: [i32; 2], kh: usize, kw: usize, sh: u32, act: bool) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        use core::arch::wasm32::*;
        let taps = kh * kw;
        // Per tap, lanes alternate (w_c0, w_c1), matching the interleaved (c0, c1) pixel pairs.
        let wv: Vec<v128> = (0..taps).map(|t| i32x4_splat((w[t] as u16 as i32) | ((w[taps + t] as i32) << 16))).collect();
        let bias = i32x4(b[0], b[1], b[0], b[1]);
        let half = i32x4_splat(1 << (sh - 1));
        let zero = i32x4_splat(0);
        for oy in 0..so.h {
            let mut ox = 0;
            while ox < so.w {
                let (mut lo, mut hi) = (bias, bias);
                for ky in 0..kh {
                    for kx in 0..kw {
                        // SAFETY: the plane has 8 pixels of slack past every row, and rows past the image are zeros.
                        let v = unsafe { v128_load(pad.as_ptr().add(((oy + ky) * pw_ + ox + kx) * 2) as *const v128) };
                        lo = i32x4_add(lo, i32x4_extmul_low_i16x8(v, wv[ky * kw + kx]));
                        hi = i32x4_add(hi, i32x4_extmul_high_i16x8(v, wv[ky * kw + kx]));
                    }
                }
                let mut l = i32x4_shr(i32x4_add(lo, half), sh);
                let mut h = i32x4_shr(i32x4_add(hi, half), sh);
                if act {
                    l = i32x4_max(l, zero);
                    h = i32x4_max(h, zero);
                }
                // A store at a row end spills into the next row's first pixels, which that row then overwrites; the
                // last row spills into the plane's slack.
                // SAFETY: (oy * w + ox + 3) < hwp thanks to the slack.
                unsafe { v128_store(y.as_mut_ptr().add(cp * so.hwp * 2 + (oy * so.w + ox) * 2) as *mut v128, i16x8_narrow_i32x4(l, h)) };
                ox += 4;
            }
        }
        true
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// 1x1 conv, stride 1. Weights [oc][ic], ic even; output channels in blocks of 4.
#[allow(clippy::too_many_arguments)]
fn pw(x: &[i16], si: Shape, y: &mut [i16], so: Shape, w: &[i16], b: &[i32], sh: u32, act: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        pw_simd(x, si, y, so, w, b, sh, act)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        pw_scalar(x, si, y, so, w, b, sh, act)
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
fn pw_scalar(x: &[i16], si: Shape, y: &mut [i16], so: Shape, w: &[i16], b: &[i32], sh: u32, act: bool) {
    for oc in 0..so.c {
        for p in 0..so.hwp {
            let mut acc = b[oc];
            for ic in 0..si.c {
                acc += w[oc * si.c + ic] as i32 * x[(ic >> 1) * si.hwp * 2 + p * 2 + (ic & 1)] as i32;
            }
            let v = sat(rs(acc, sh));
            y[(oc >> 1) * so.hwp * 2 + p * 2 + (oc & 1)] = if act && v < 0 { 0 } else { v as i16 };
        }
    }
}

/// 4 output channels x 8 pixels per step: two loads of (4 px x 2 ch), four weight pairs, eight dot products.
#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn pw_simd(x: &[i16], si: Shape, y: &mut [i16], so: Shape, w: &[i16], b: &[i32], sh: u32, act: bool) {
    use core::arch::wasm32::*;
    let icp = si.c / 2;
    let plane_i = si.hwp * 2;
    let plane_o = so.hwp * 2;
    let half = i32x4_splat(1 << (sh - 1));
    let zero = i32x4_splat(0);
    // Weight pairs as i32: (w[oc][2j], w[oc][2j+1]) little-endian, matching dot_i16x8's lane pairing.
    let wp = |oc: usize, j: usize| -> i32 { (w[oc * si.c + 2 * j] as u16 as i32) | ((w[oc * si.c + 2 * j + 1] as i32) << 16) };
    let mut oc = 0;
    while oc < so.c {
        let wpairs: Vec<[i32; 4]> = (0..icp).map(|j| [wp(oc, j), wp(oc + 1, j), wp(oc + 2, j), wp(oc + 3, j)]).collect();
        let bias = [i32x4_splat(b[oc]), i32x4_splat(b[oc + 1]), i32x4_splat(b[oc + 2]), i32x4_splat(b[oc + 3])];
        let mut p = 0;
        while p < so.hwp {
            let mut a = [bias[0], bias[1], bias[2], bias[3], bias[0], bias[1], bias[2], bias[3]];
            for (j, wq) in wpairs.iter().enumerate() {
                // SAFETY: p + 7 < hwp and j < c/2, so both loads are inside the plane (bounds validated at load).
                let (v0, v1) = unsafe {
                    let base = x.as_ptr().add(j * plane_i + p * 2);
                    (v128_load(base as *const v128), v128_load(base.add(8) as *const v128))
                };
                for q in 0..4 {
                    let ws = i32x4_splat(wq[q]);
                    a[q] = i32x4_add(a[q], i32x4_dot_i16x8(v0, ws));
                    a[4 + q] = i32x4_add(a[4 + q], i32x4_dot_i16x8(v1, ws));
                }
            }
            for half_i in 0..2 {
                let mut r = [zero; 4];
                for q in 0..4 {
                    let v = i32x4_shr(i32x4_add(a[half_i * 4 + q], half), sh);
                    r[q] = if act { i32x4_max(v, zero) } else { v };
                }
                // Interleave channels (oc, oc+1) and (oc+2, oc+3) per pixel, narrowing with int16 saturation.
                let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r[0], r[1]);
                let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r[0], r[1]);
                let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r[2], r[3]);
                let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r[2], r[3]);
                let px = p + half_i * 4;
                // SAFETY: px + 3 < hwp and oc + 3 < c: inside the output planes.
                unsafe {
                    v128_store(y.as_mut_ptr().add((oc >> 1) * plane_o + px * 2) as *mut v128, i16x8_narrow_i32x4(lo01, hi01));
                    v128_store(y.as_mut_ptr().add(((oc >> 1) + 1) * plane_o + px * 2) as *mut v128, i16x8_narrow_i32x4(lo23, hi23));
                }
            }
            p += 8;
        }
        oc += 4;
    }
}

// ---- C ABI for JS ---------------------------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn alloc_bytes(n: usize) -> *mut u8 {
    let mut v: Vec<u8> = Vec::with_capacity(n.max(1));
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

/// # Safety
/// The pointers must reference `nwords` u32s, `nw` i16s and `nb` i32s inside linear memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prog_load(words: *const u32, nwords: usize, w: *const i16, nw: usize, b: *const i32, nb: usize) -> *mut Prog {
    let (words, w, b) = unsafe { (core::slice::from_raw_parts(words, nwords), core::slice::from_raw_parts(w, nw).to_vec(), core::slice::from_raw_parts(b, nb).to_vec()) };
    match Prog::parse(words, w, b) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => core::ptr::null_mut(),
    }
}

/// # Safety
/// `p` must come from `prog_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prog_input(p: *mut Prog) -> *mut i16 {
    unsafe { (*p).input_mut().as_mut_ptr() }
}

/// # Safety
/// `p` must come from `prog_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prog_run(p: *mut Prog) {
    unsafe { (*p).run() }
}

/// # Safety
/// `p` must come from `prog_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prog_sums(p: *mut Prog) -> *const i32 {
    unsafe { (*p).sums().as_ptr() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Vec<u32>, Vec<i16>, Vec<i32>) {
        // input 4ch 2x2 -> pw to 4ch -> gap
        let words = vec![MAGIC, 1, 2, 2, 0, 4, 4, 2, 2, 4, 2, 2, 1, 1, 0, 1, 1, 0, 0, 0, 6, 1, 0];
        let w: Vec<i16> = (0..16).map(|i| (i % 5) as i16 - 2).collect();
        (words, w, vec![1, -1, 2, 0])
    }

    #[test]
    fn rejects_garbage() {
        let (mut words, w, b) = tiny();
        assert!(Prog::parse(&words, w.clone(), b.clone()).is_some());
        words[12] = 9; // out-of-range tensor id
        assert!(Prog::parse(&words, w.clone(), b.clone()).is_none());
        let (words, _, b) = tiny();
        assert!(Prog::parse(&words, vec![0; 3], b).is_none()); // weights too short
        assert!(Prog::parse(&words[..10], vec![0; 16], vec![0; 4]).is_none()); // truncated
    }

    #[test]
    fn pw_matches_definition() {
        let (words, w, b) = tiny();
        let mut p = Prog::parse(&words, w.clone(), b.clone()).unwrap();
        let inp: Vec<i16> = (0..12).map(|i| i * 300 - 1700).collect();
        p.input_mut().copy_from_slice(&inp);
        p.run();
        for oc in 0..4 {
            let mut gap = 0;
            for px in 0..4 {
                let mut acc = b[oc];
                for ic in 0..3 {
                    acc += w[oc * 4 + ic] as i32 * inp[px * 3 + ic] as i32;
                }
                gap += sat(rs(acc, 1));
            }
            assert_eq!(p.sums()[oc], gap);
        }
    }
}

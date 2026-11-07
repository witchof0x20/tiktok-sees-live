// Video frame -> rgba8 texture with on-demand box-filtered mips, and per-face crop parameters that reproduce
// upstream's drawCrop (similarity transform placing eye and mouth midpoints on the net's template).

export const EXP = { S: 256, E: [126.63, 75.15], M: [127.65, 179.75] };
export const AGE = { S: 224, E: [111.92, 75.01], M: [112.63, 148.25] };
const MAX_LEVEL = 6;

export function transform(t, e, m) {
  const dx = m[0] - e[0], dy = m[1] - e[1], Dx = t.M[0] - t.E[0], Dy = t.M[1] - t.E[1], n = dx * dx + dy * dy;
  const re = (Dx * dx + Dy * dy) / n, im = (Dy * dx - Dx * dy) / n;
  return { re, im, tx: t.E[0] - (re * e[0] - im * e[1]), ty: t.E[1] - (im * e[0] + re * e[1]) };
}

const levelSize = (w, h, l) => { for (let i = 0; i < l; i++) { w = Math.max(1, w >> 1); h = Math.max(1, h >> 1); } return [w, h]; };

// Inverse of the canvas transform drawCrop sets (k = 1): crop pixel center -> source texel coordinates in the
// chosen mip level. Same level choice as upstream: halve until the scale is at least 1/4.
export function cropParams(t, w, h, e, m) {
  const f = transform(t, e, m);
  let s = Math.hypot(f.re, f.im), lvl = 0;
  while (s < 0.25 && lvl < MAX_LEVEL) { s *= 2; lvl++; }
  const [lw] = levelSize(w, h, lvl), scale = w / lw, re = f.re * scale, im = f.im * scale;
  const E = f.tx + 0.5 - (re - im) * 0.5, Fy = f.ty + 0.5 - (im + re) * 0.5, det = re * re + im * im;
  return { m: [re / det, -im / det, im / det, re / det], t: [-(re * E + im * Fy) / det, (im * E - re * Fy) / det], lvl };
}

const MIP = `@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = textureDimensions(dst);
  if (g.x >= d.x || g.y >= d.y) { return; }
  let s = vec2<i32>(g.xy) * 2; let m = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
  let c = textureLoad(src, min(s, m), 0) + textureLoad(src, min(s + vec2<i32>(1, 0), m), 0)
        + textureLoad(src, min(s + vec2<i32>(0, 1), m), 0) + textureLoad(src, min(s + vec2<i32>(1, 1), m), 0);
  textureStore(dst, g.xy, c * 0.25);
}`;

export class FrameTexture {
  constructor(device) {
    this.device = device;
    this.tex = null;
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: MIP }), entryPoint: 'main' } });
  }
  upload(source, w, h) {
    const d = this.device;
    if (!this.tex || this.tex.width !== w || this.tex.height !== h) {
      this.tex?.destroy();
      const levels = Math.min(MAX_LEVEL, Math.floor(Math.log2(Math.max(w, h)))) + 1;
      this.tex = d.createTexture({ size: [w, h], format: 'rgba8unorm', mipLevelCount: levels, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
      this.view = this.tex.createView();
      this.mipBind = [];
      for (let l = 1; l < levels; l++) this.mipBind.push(d.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.tex.createView({ baseMipLevel: l - 1, mipLevelCount: 1 }) },
        { binding: 1, resource: this.tex.createView({ baseMipLevel: l, mipLevelCount: 1 }) }] }));
    }
    if (!FrameTexture.viaCanvas) {
      try { d.queue.copyExternalImageToTexture({ source }, { texture: this.tex }, [w, h]); return; }
      catch (err) {
        // Firefox only takes bitmaps and canvases here, not <video> or VideoFrame: go through a 2D canvas from now on.
        if (!(err instanceof TypeError)) throw err;
        FrameTexture.viaCanvas = true;
      }
    }
    const c = FrameTexture.staging ??= new OffscreenCanvas(w, h);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    c.getContext('2d').drawImage(source, 0, 0, w, h);
    d.queue.copyExternalImageToTexture({ source: c }, { texture: this.tex }, [w, h]);
  }
  // Builds levels 1..upTo; only needed when a face is small enough that the crop samples a reduced level.
  mips(enc, upTo) {
    if (upTo < 1) return;
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    for (let l = 1; l <= Math.min(upTo, this.mipBind.length); l++) {
      const [w, h] = levelSize(this.tex.width, this.tex.height, l);
      pass.setBindGroup(0, this.mipBind[l - 1]);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    }
    pass.end();
  }
}

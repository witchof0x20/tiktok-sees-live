// Draws a frame texture to a WebGPU canvas (optionally mirrored). Replaces <video> for the preview: on some
// Linux/Vulkan Chrome setups a <video> element shows camera frames as blank, while VideoFrames are fine.
const SHADER = `struct U { mirror: f32, pad: vec3<f32> }
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> u: U;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  var o: V;
  o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2<f32>(select(p.x, 1.0 - p.x, u.mirror > 0.5), 1.0 - p.y);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> { return vec4<f32>(textureSampleLevel(tex, smp, v.uv, 0.0).rgb, 1.0); }`;

export class Display {
  constructor(device, canvas) {
    this.device = device; this.canvas = canvas;
    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    const module = device.createShaderModule({ code: SHADER });
    this.pipeline = device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] }, primitive: { topology: 'triangle-list' } });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cache = new Map();
  }
  draw(enc, view, mirror) {
    this.device.queue.writeBuffer(this.uniform, 0, new Float32Array([mirror ? 1 : 0, 0, 0, 0]));
    let bg = this.cache.get(view);
    if (!bg) {
      if (this.cache.size > 16) this.cache.clear();
      bg = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: view }, { binding: 1, resource: this.sampler }, { binding: 2, resource: { buffer: this.uniform } }] });
      this.cache.set(view, bg);
    }
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  }
}

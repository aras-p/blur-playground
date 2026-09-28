/*
 * Skia-style image blur, based on Google Skia da51f0d60e:
 * src/core/SkImageFilterTypes.cpp (rescale), SkBlurEngine.cpp (Gaussian kernels).
 * Copyright 2023 Google LLC. BSD-3-Clause; see skia-LICENSE.txt.
 * Simplified to a whole image with clamped edges, no crop/transform/tile modes.
 */
class SkiaGaussian {
    constructor(device, vertexShader, beginPass) {
        this.device = device;
        this.beginPass = beginPass;
        this.textures = [];
        this.buffers = [];
        this.sampler = device.createSampler({minFilter: 'linear', magFilter: 'linear'});
        const common = vertexShader + `
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
struct Params {
    mapping: vec4f, // source pixel = output pixel * xy + zw
    axis: vec4f,    // blur direction, sample count, border flag
    taps: array<vec4f, 14>,
};
@group(0) @binding(2) var<uniform> p: Params;
fn samplePixel(pos: vec2f) -> vec4f {
    return textureSampleLevel(tex, smp, pos / vec2f(textureDimensions(tex)), 0.0);
}
`;
        const pipeline = body => device.createRenderPipeline({
            layout: 'auto',
            vertex: {module: device.createShaderModule({code: vertexShader}), entryPoint: 'vs_main'},
            fragment: {module: device.createShaderModule({code: common + body}),
                entryPoint: 'fs_main', targets: [{format: 'rgba32float'}]},
            primitive: {topology: 'triangle-list'},
        });
        this.resize = pipeline(`
@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    var src = pos.xy * p.mapping.xy + p.mapping.zw;
    // Preserve original edge/corner samples in the one-pixel padding at each
    // reduction, rather than clamping to a filtered interior pixel.
    if (p.axis.w != 0.0) {
        let size = vec2f(textureDimensions(tex));
        if (pos.x < 1.0) { src.x = 0.5; }
        if (pos.y < 1.0) { src.y = 0.5; }
        if (pos.x > p.axis.x - 1.0) { src.x = size.x - 0.5; }
        if (pos.y > p.axis.y - 1.0) { src.y = size.y - 0.5; }
    }
    return samplePixel(src);
}`);
        this.convolve = pipeline(`
@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    var color = vec4f(0.0);
    for (var i = 0u; i < u32(p.axis.z); i++) {
        let tap = p.taps[i];
        color += tap.y * samplePixel(pos.xy + p.axis.xy * tap.x);
    }
    return color;
}`);
        this.convolve2D = pipeline(`
@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    var color = vec4f(0.0);
    for (var y = 0u; y < u32(p.axis.y); y++) {
        for (var x = 0u; x < u32(p.axis.x); x++) {
            color += p.taps[x].y * p.taps[y].w *
                samplePixel(pos.xy + vec2f(p.taps[x].x, p.taps[y].z));
        }
    }
    return color;
}`);
    }

    static steps(scale) {
        let n = Math.ceil(Math.log2(Math.ceil(1 / scale)));
        if (n > 0 && scale * 2 ** (n - 1) >= (n === 1 ? 0.999 : 0.9)) n--;
        return n;
    }

    static kernel(sigma) {
        if (sigma <= 0.03) return [[0, 1]];
        const radius = Math.ceil(Math.fround(3 * sigma));
        const weights = Array.from({length: radius + 1}, (_, i) => Math.exp(-0.5 * (i / sigma) ** 2));
        const sum = weights[0] + 2 * weights.slice(1).reduce((a, b) => a + b, 0);
        return weights.map((w, i) => [i, w / sum]);
    }

    static linearKernel(sigma) {
        const weights = SkiaGaussian.kernel(sigma).map(t => t[1]);
        const radius = weights.length - 1;
        const taps = [];
        let i = 1;
        if (radius % 2) {
            const w = weights[0] / 2 + weights[1];
            const offset = weights[1] / w;
            taps.push([-offset, w], [offset, w]);
            i = 2;
        } else taps.push([0, weights[0]]);
        for (; i <= radius; i += 2) {
            const w = weights[i] + weights[i + 1];
            const offset = i + weights[i + 1] / w;
            taps.push([-offset, w], [offset, w]);
        }
        return taps.sort((a, b) => a[0] - b[0]);
    }

    texture(width, height) {
        const index = this.textureIndex++;
        let tex = this.textures[index];
        if (!tex || tex.width !== width || tex.height !== height) {
            tex?.destroy();
            tex = this.device.createTexture({size: [width, height], format: 'rgba32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT |
                       GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST});
            this.textures[index] = tex;
        }
        return tex;
    }

    pass(encoder, pipeline, input, output, data) {
        const index = this.bufferIndex++;
        const buffer = this.buffers[index] ??= this.device.createBuffer({size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
        this.device.queue.writeBuffer(buffer, 0, data);
        const pass = this.beginPass(encoder, {colorAttachments: [{view: output.createView(),
            loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0]}]});
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
            entries: [{binding: 0, resource: this.sampler}, {binding: 1, resource: input.createView()},
                      {binding: 2, resource: {buffer}}]}));
        pass.draw(3);
        pass.end();
    }

    blur(encoder, input, output, radiusX, radiusY, intermediates = []) {
        this.textureIndex = this.bufferIndex = 0;
        let sigma = [radiusX, radiusY].map(r => Math.fround(Math.min(532, Math.max(0, r / 3))));
        sigma = sigma.map(s => s <= 0.03 ? 0 : s);
        const scale = sigma.map(s => s > 4 ? Math.fround(4 / s) : 1);
        const steps = scale.map(SkiaGaussian.steps);
        const finalScale = scale.map((s, a) => steps[a] ? s : 1);
        const size = [input.width, input.height];
        let bounds = [0, 0, ...size]; // fractional logical bounds: left, top, right, bottom
        let origin = [0, 0]; // integer pixel origin, including padding
        let tex = input;
        while (steps.some(n => n > 0)) {
            const next = bounds.slice();
            const factor = [1, 1];
            for (let a = 0; a < 2; a++) {
                if (!steps[a]) continue;
                factor[a] = steps[a] > 1 ? 0.5 : size[a] * finalScale[a] / (bounds[a + 2] - bounds[a]);
                steps[a]--;
                const center = (bounds[a] + bounds[a + 2]) / 2;
                next[a] = (bounds[a] - center) * factor[a];
                next[a + 2] = (bounds[a + 2] - center) * factor[a];
            }
            // Skia's tolerant roundOut avoids an extra pixel from float error.
            const lo = [Math.floor(next[0] + 0.001) - 1, Math.floor(next[1] + 0.001) - 1];
            const hi = [Math.ceil(next[2] - 0.001) + 1, Math.ceil(next[3] - 0.001) + 1];
            const dst = this.texture(hi[0] - lo[0], hi[1] - lo[1]);
            const data = new Float32Array(64);
            for (let a = 0; a < 2; a++) {
                data[a] = 1 / factor[a];
                data[a + 2] = (lo[a] - next[a]) / factor[a] + bounds[a] - origin[a];
            }
            data.set([dst.width, dst.height, 0, 1], 4);
            this.pass(encoder, this.resize, tex, dst, data);
            tex = dst; bounds = next; origin = lo;
            intermediates.push({texture: tex, name: `Skia downsample: ${tex.width}×${tex.height}`});
        }
        const reduced = tex !== input;
        sigma = sigma.map((s, a) => Math.min(4, s * finalScale[a]));
        const kernels = sigma.map(SkiaGaussian.kernel);
        const radii = kernels.map(k => k.length - 1);
        if (radii[0] && radii[1] && (2 * radii[0] + 1) * (2 * radii[1] + 1) <= 28) {
            const data = new Float32Array(64);
            for (let a = 0; a < 2; a++) {
                const r = radii[a];
                data[4 + a] = 2 * r + 1;
                for (let i = -r; i <= r; i++) {
                    data[8 + (i + r) * 4 + a * 2] = i;
                    data[9 + (i + r) * 4 + a * 2] = kernels[a][Math.abs(i)][1];
                }
            }
            const dst = reduced ? this.texture(tex.width, tex.height) : output;
            this.pass(encoder, this.convolve2D, tex, dst, data);
            tex = dst;
            intermediates.push({texture: tex, name: `Skia Gaussian 2D: ${tex.width}×${tex.height}`});
        } else {
            for (let a = 0; a < 2; a++) {
                if (!radii[a]) continue;
                const data = new Float32Array(64);
                const taps = SkiaGaussian.linearKernel(sigma[a]);
                data[4 + a] = 1; data[6] = taps.length;
                taps.forEach((tap, i) => data.set(tap, 8 + i * 4));
                const lastAxis = a === 1 || !radii[1];
                const dst = !reduced && lastAxis ? output : this.texture(tex.width, tex.height);
                this.pass(encoder, this.convolve, tex, dst, data);
                tex = dst;
                intermediates.push({texture: tex, name: `Skia Gaussian ${a ? 'Y' : 'X'}: ${tex.width}×${tex.height}`});
            }
        }
        const data = new Float32Array(64);
        data.set([(bounds[2] - bounds[0]) / size[0], (bounds[3] - bounds[1]) / size[1],
                  bounds[0] - origin[0], bounds[1] - origin[1]]);
        if (reduced) this.pass(encoder, this.resize, tex, output, data);
        else if (tex !== output) encoder.copyTextureToTexture(
            {texture: tex}, {texture: output}, size);
        // Bound memory while animation visits many different fractional scales.
        while (this.textures.length > this.textureIndex) this.textures.pop().destroy();
    }

    destroy() {
        this.textures.forEach(t => t.destroy());
        this.buffers.forEach(b => b.destroy());
        this.textures = []; this.buffers = [];
    }
}

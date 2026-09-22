/* SPDX-FileCopyrightText: 2024 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Port of Blender compositor recursive_gaussian_blur.cc,
 * cached_resources/intern/{deriche,van_vliet}_gaussian_coefficients.cc and
 * shaders/compositor_{deriche,van_vliet}_gaussian_blur{,_sum}.glsl.
 * Coefficients are evaluated in double precision, then uploaded as floats.
 */

function fastGaussianDeriche(sigma) {
    const a0 = 1.6797292232361107, a1 = 3.7348298269103580;
    const b0 = 1.7831906544515104, b1 = 1.7228297663338028;
    const c0 = -0.6802783501806897, c1 = -0.2598300478959625;
    const w0 = 0.6318113174569493, w1 = 1.9969276832487770;
    const e0 = Math.exp(-b0 / sigma), e1 = Math.exp(-b1 / sigma);
    const cos0 = Math.cos(w0 / sigma), cos1 = Math.cos(w1 / sigma);
    const sin0 = Math.sin(w0 / sigma), sin1 = Math.sin(w1 / sigma);
    let causal = [a0 + c0,
        e1 * (c1 * sin1 - (c0 + 2 * a0) * cos1) + e0 * (a1 * sin0 - (2 * c0 + a0) * cos0),
        2 * e0 * e1 * ((a0 + c0) * cos1 * cos0 - cos1 * a1 * sin0 - cos0 * c1 * sin1)
            + c0 * e0 * e0 + a0 * e1 * e1,
        e1 * e0 * e0 * (c1 * sin1 - cos1 * c0) + e0 * e1 * e1 * (a1 * sin0 - cos0 * a0)];
    const feedback = [-2 * e0 * cos0 - 2 * e1 * cos1,
        4 * cos1 * cos0 * e0 * e1 + e1 * e1 + e0 * e0,
        -2 * cos0 * e0 * e1 * e1 - 2 * cos1 * e1 * e0 * e0,
        e0 * e0 * e1 * e1];
    const sum = xs => xs.reduce((a, b) => a + b, 0);
    // Blender's normalization helper returns float, although its inputs are doubles.
    const normalization = Math.fround(2 * sum(causal) / (1 + sum(feedback)) - causal[0]);
    causal = causal.map(v => v / normalization);
    const nonCausal = feedback.map((v, i) => (causal[i + 1] || 0) - v * causal[0]);
    return [causal, nonCausal].map(feedforward => ({feedforward, feedback,
        boundary: sum(feedforward) / (1 + sum(feedback))}));
}

function fastGaussianVanVliet(sigma) {
    const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
    const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const div = (a, b) => {
        const d = b[0] * b[0] + b[1] * b[1];
        return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
    };
    const polar = (r, p) => [r * Math.cos(p), r * Math.sin(p)];
    const one = [1, 0];
    const poles = [[1.12075, 1.27788], [1.12075, -1.27788], [1.76952, 0.46611], [1.76952, -0.46611]];
    let scale = sigma / 2;
    for (let i = 0; i < 10; i++) {
        let variance = 0, derivative = 0;
        for (const p of poles) {
            const abs = Math.hypot(...p), arg = Math.atan2(p[1], p[0]);
            const magnitude = Math.pow(abs, 1 / scale), phase = arg / scale;
            const m1 = polar(magnitude, phase), unit = polar(1, phase);
            const delta = sub([magnitude, 0], unit);
            variance += mul([2, 0], div(m1, mul(delta, delta)))[0];
            derivative += div(mul(mul(mul([2, 0], m1), add([magnitude, 0], unit)),
                [Math.log(abs), -arg]), mul(mul(mul(delta, delta), delta), [scale * scale, 0]))[0];
        }
        if (Math.abs(sigma * sigma - variance) < 1e-7) break;
        scale -= (variance - sigma * sigma) / derivative;
    }
    const nonCausal = poles.map(p => polar(Math.pow(Math.hypot(...p), 1 / scale), Math.atan2(p[1], p[0]) / scale));
    const causal = nonCausal.map(p => div(one, p));
    // Equation (13), with the same summation order as Blender.
    let gain = one;
    for (const p of nonCausal) gain = div(gain, p);
    const [p0, p1, p2, p3] = nonCausal;
    const b4 = gain;
    const b3 = mul([-gain[0], -gain[1]], add(add(add(p0, p1), p2), p3));
    const b2 = mul(gain, [mul(p1,p0), mul(p2,p0), mul(p2,p1), mul(p3,p0), mul(p3,p1), mul(p3,p2)].reduce(add));
    const b1 = mul([-gain[0], -gain[1]], [mul(mul(p2,p1),p0), mul(mul(p3,p1),p0),
        mul(mul(p3,p2),p0), mul(mul(p3,p2),p1)].reduce(add));
    const feedforward = 1 + (b1[0] + b2[0] + b3[0] + b4[0]);
    return [0, 2].flatMap(index => {
        const pole = causal[index], inverse = div(one, pole);
        let residue = one, transfer = one;
        for (let j = 0; j < 4; j++) {
            if (Math.floor(j / 2) !== index / 2) residue = mul(residue, sub(one, mul(causal[j], inverse)));
            transfer = mul(transfer, sub(one, mul(causal[j], pole)));
        }
        const parallel = mul(div([feedforward, 0], residue), div([feedforward, 0], transfer));
        const feedback = [-2 * pole[0], pole[0] * pole[0] + pole[1] * pole[1]];
        const c1 = parallel[1] / inverse[1], c0 = parallel[0] - c1 * inverse[0];
        return [[c0, c1], [c1 - c0 * feedback[0], -c0 * feedback[1]]].map(ff => ({
            feedforward: ff, feedback, boundary: (ff[0] + ff[1]) / (1 + feedback[0] + feedback[1]),
        }));
    });
}

// Each invocation scans one row in one direction. Independent causal/non-causal
// sections write separate images; the sum pass transposes for the next axis.
function fastGaussianShader(order, count) {
    return `
struct Section { feedforward: vec4f, feedback: vec4f, boundary: vec4f }
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<uniform> sections: array<Section, ${count}>;
${Array.from({length: count}, (_, i) => `@group(0) @binding(${i + 2}) var output${i}: texture_storage_2d<rgba32float, write>;`).join('\n')}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(input);
    if (id.x >= size.y) { return; }
    let causal = id.y % 2u == 0u;
    let section = sections[id.y];
    let edge = textureLoad(input, vec2i(select(i32(size.x) - 1, 0, causal), i32(id.x)), 0);
    var inputs: array<vec4f, ${order + 1}>;
    var outputs: array<vec4f, ${order + 1}>;
    for (var i = 0; i <= ${order}; i++) {
        inputs[i] = edge;
        outputs[i] = edge * section.boundary.x;
    }
    for (var x = 0; x < i32(size.x); x++) {
        let texel = vec2i(select(i32(size.x) - 1 - x, x, causal), i32(id.x));
        inputs[0] = textureLoad(input, texel, 0);
        outputs[0] = vec4f(0);
        let first = select(1, 0, causal);
        for (var i = 0; i < ${order}; i++) {
            outputs[0] += section.feedforward[i] * inputs[first + i];
            outputs[0] -= section.feedback[i] * outputs[i + 1];
        }
        switch id.y {
${Array.from({length: count}, (_, i) => `case ${i}u: { textureStore(output${i}, texel, outputs[0]); }`).join('\n')}
            default: {}
        }
        for (var i = ${order}; i >= 1; i--) { inputs[i] = inputs[i - 1]; outputs[i] = outputs[i - 1]; }
    }
}`;
}

function fastGaussianSumShader(count) {
    return `
${Array.from({length: count}, (_, i) => `@group(0) @binding(${i}) var input${i}: texture_2d<f32>;`).join('\n')}
@group(0) @binding(${count}) var output: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= textureDimensions(input0))) { return; }
    let p = vec2i(id.xy);
    let color = ${Array.from({length: count}, (_, i) => `textureLoad(input${i}, p, 0)`).join(' + ')};
    textureStore(output, p.yx, color);
}`;
}

const fastGaussianPipelines = new Map();
const fastGaussianUniforms = [];
function fastGaussianPass(encoder, input, output, sigma, vanVliet, axis) {
    const count = vanVliet ? 4 : 2;
    if (!fastGaussianPipelines.has(count)) {
        const pipeline = code => gpu_device.createComputePipeline({layout: 'auto',
            compute: {module: gpu_device.createShaderModule({code}), entryPoint: 'main'}});
        fastGaussianPipelines.set(count, [pipeline(fastGaussianShader(vanVliet ? 2 : 4, count)),
            pipeline(fastGaussianSumShader(count))]);
    }
    const [filter, sum] = fastGaussianPipelines.get(count);
    const sections = vanVliet ? fastGaussianVanVliet(sigma) : fastGaussianDeriche(sigma);
    const data = new Float32Array(48);
    sections.forEach((s, i) => {
        data.set(s.feedforward, i * 12); data.set(s.feedback, i * 12 + 4); data[i * 12 + 8] = s.boundary;
    });
    if (!fastGaussianUniforms[axis]) fastGaussianUniforms[axis] = gpu_device.createBuffer({
        size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    gpu_device.queue.writeBuffer(fastGaussianUniforms[axis], 0, data);
    const results = Array.from({length: count}, () => getCachedTexture(input.width, input.height, GPUTextureUsage.STORAGE_BINDING));
    let pass = beginBlurPass(encoder, {}, true);
    pass.setPipeline(filter);
    pass.setBindGroup(0, gpu_device.createBindGroup({layout: filter.getBindGroupLayout(0), entries: [
        {binding: 0, resource: input.createView()}, {binding: 1, resource: {buffer: fastGaussianUniforms[axis]}},
        ...results.map((t, i) => ({binding: i + 2, resource: t.createView()})),
    ]}));
    pass.dispatchWorkgroups(Math.ceil(input.height / 64), count); pass.end();
    pass = beginBlurPass(encoder, {}, true);
    pass.setPipeline(sum);
    pass.setBindGroup(0, gpu_device.createBindGroup({layout: sum.getBindGroupLayout(0), entries: [
        ...results.map((t, i) => ({binding: i, resource: t.createView()})),
        {binding: count, resource: output.createView()},
    ]}));
    pass.dispatchWorkgroups(Math.ceil(input.width / 16), Math.ceil(input.height / 16)); pass.end();
}

function blurFastGaussian(encoder, input, output) {
    const rx = blur_params.radius_x, ry = blur_params.radius_y;
    if (rx === 0 && ry === 0) {
        encoder.copyTextureToTexture({texture: input}, {texture: output}, [input.width, input.height]);
        return;
    }
    const sx = Math.fround(Math.max(1, rx) / 3), sy = Math.fround(Math.max(1, ry) / 3);
    const largest = Math.max(sx, sy);
    if (largest < 3) {
        const tmp = getCachedTexture(input.width, input.height);
        separablePass(encoder, input, tmp, BlurMode.GAUSSIAN, true, rx);
        separablePass(encoder, tmp, output, BlurMode.GAUSSIAN, false, ry);
        intermediateTextures.push({texture: tmp, name: 'Fast Gaussian: horizontal (direct)'});
        return;
    }
    const tmp = getCachedTexture(input.height, input.width, GPUTextureUsage.STORAGE_BINDING);
    fastGaussianPass(encoder, input, tmp, sx, largest >= 32, 0);
    fastGaussianPass(encoder, tmp, output, sy, largest >= 32, 1);
    intermediateTextures.push({texture: tmp, name: `Fast Gaussian: horizontal, transposed (${largest >= 32 ? 'Van Vliet' : 'Deriche'})`});
}

/* "Ryg Blur": repeated box convolution, following Fabian Giesen's Fast blurs 1/2:
 * https://fgiesen.wordpress.com/2012/07/30/fast-blurs-1/
 * https://fgiesen.wordpress.com/2012/08/01/fast-blurs-2/
 *
 * Fixed cost independent of blur size, with two bilinear samples for
 * fractional box endpoint updates. Iterations control how many times to do the
 * convolution: 1 gives a box filter, 2 a tent filter, 3 and up approach Gaussian.
 *
 * One invocation walks a scanline; parallelism is only over rows or columns,
 * and multiple passes over the full size image incur a lot of memory traffic. */

let pip_ryg = null;
const rygBuffers = [];

function initRygBlur() {
    pip_ryg = gpu_device.createComputePipeline({
        layout: 'auto',
        compute: { module: gpu_device.createShaderModule({ code: RYG_SHADER }), entryPoint: 'main' },
    });
}

// Match variance to sigma = UI radius / 3, split across the box passes.
// For an integer radius m the variance is m(m+1)/3. Solve for the
// fractional endpoint weight exactly, keeping strength stable as count changes.
function rygBoxRadius(radius, count) {
    const variance = (radius / 3) ** 2 / count;
    const m = Math.floor((Math.sqrt(12 * variance + 1) - 1) / 2);
    const alpha = (2 * m + 1) * (variance - m * (m + 1) / 3) /
        (2 * ((m + 1) ** 2 - variance));
    return m + Math.max(0, Math.min(1, alpha));
}

function blurRyg(commandEncoder, input, output) {
    const count = blur_params.ryg_box_count;
    const axes = [blur_params.radius_x, blur_params.radius_y];
    const passCount = axes.filter(r => r > 0).length * count;
    let source = input, passIndex = 0;
    for (let axis = 0; axis < 2; ++axis) {
        if (axes[axis] <= 0) continue;
        const radius = rygBoxRadius(axes[axis], count);
        for (let box = 0; box < count; ++box) {
            const last = passIndex === passCount - 1;
            const target = last ? output : getCachedTexture(input.width, input.height, GPUTextureUsage.STORAGE_BINDING);
            const buffer = rygBuffers[passIndex] ??= gpu_device.createBuffer({
                size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const data = new ArrayBuffer(16), view = new DataView(data);
            view.setFloat32(0, radius, true);
            view.setUint32(4, axis, true);
            gpu_device.queue.writeBuffer(buffer, 0, data);
            const bindGroup = gpu_device.createBindGroup({
                layout: pip_ryg.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: gpu_sampler_linear },
                    { binding: 1, resource: source.createView() },
                    { binding: 2, resource: target.createView() },
                    { binding: 3, resource: { buffer } },
                ],
            });
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pip_ryg);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil((axis === 0 ? input.height : input.width) / 64));
            pass.end();
            if (!last) intermediateTextures.push({ texture: target, name: `Ryg ${axis === 0 ? 'X' : 'Y'} box ${box + 1}` });
            source = target;
            ++passIndex;
        }
    }
}

const RYG_SHADER = `
struct Params { radius: f32, axis: u32 }
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: Params;
// Exact loads initialize the sum; hardware filtering handles the updates.
fn loadLine(p: i32, line: u32) -> vec4f {
    let dims = vec2i(textureDimensions(src));
    let pos = select(vec2i(p, i32(line)), vec2i(i32(line), p), params.axis == 1u);
    return textureLoad(src, clamp(pos, vec2i(0), dims - 1), 0);
}
// Sampler subtexel precision can introduce recurrence errors at small radii.
fn sampleLine(p: f32, line: u32) -> vec4f {
    let pos = select(vec2f(p, f32(line)), vec2f(f32(line), p), params.axis == 1u);
    return textureSampleLevel(src, smp, (pos + 0.5) / vec2f(textureDimensions(src)), 0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(src);
    let lines = select(dims.y, dims.x, params.axis == 1u);
    let length = select(dims.x, dims.y, params.axis == 1u);
    if (id.x >= lines) { return; }
    let m = i32(floor(params.radius));
    let alpha = fract(params.radius);
    let scale = 1.0 / (2.0 * params.radius + 1.0);
    // Clamp-to-edge: the entire negative half is the first pixel.
    var sum = loadLine(0, id.x) * f32(m + 1);
    for (var k = 1; k <= m; k++) { sum += loadLine(k, id.x); }
    sum += alpha * (loadLine(0, id.x) + loadLine(m + 1, id.x));
    for (var p = 0u; p < length; p++) {
        let pos = select(vec2u(p, id.x), vec2u(id.x, p), params.axis == 1u);
        textureStore(dst, pos, sum * scale);
        let incoming = sampleLine(f32(p) + params.radius + 1.0, id.x);
        let outgoing = sampleLine(f32(p) - params.radius, id.x);
        sum += incoming - outgoing;
    }
}
`;

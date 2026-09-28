/* Smol Gaussian: a Gaussian approximation using downsampling, small separable
 * filters, and reconstruction, with smooth transitions between working resolutions.
 * Each axis chooses its working resolution independently.
 * Reduction and reconstruction already add blur, so each endpoint subtracts
 * their approximate variances before choosing its residual Gaussian sigma.
 *
 * A resolution change would otherwise change the approximation abruptly. Blend
 * neighboring grids, each computed for the SAME requested radius, before
 * retiring the finer grid. The canonical reduction tree shares work and keeps
 * each endpoint's sampling history independent of its current blend weight.
 * Reconstruction samples each endpoint directly onto the final output grid.
 *
 * Unlike Dual Kawase, these endpoints are alternative approximations of one
 * target blur, not fixed kernels of different widths. Area reduction, Gaussian
 * filtering, and bilinear/cubic reconstruction are established building blocks;
 * the variance estimates, 5–6 sigma transition, tapered tails, and triangular
 * blending are choices of this approximation, not an exact Gaussian identity.
 *
 * Related work (building blocks, not a specification of this exact combination):
 * - Fabian Giesen, "Gaussian blur kernels" (2009): prefilter before reducing,
 *   blur at lower resolution, then reconstruct; skipping samples causes aliasing.
 *   https://sourceforge.net/p/gdalgorithms/mailman/message/23077758/
 * - Cornell CS5625, "Apply blur to mipmap levels" (2022): reduce by 2^k and use
 *   sigma/2^k. Its bloom merge mixes widths, not same-target resolution choices.
 *   https://www.cs.cornell.edu/courses/cs5625/2022sp/assignments/pipeline.html#323-apply-blur-to-mipmap-levels
 * - Intel, "An Investigation of Fast Real-Time GPU-Based Image Blur Algorithms",
 *   "Working in Lower Resolution": reduce, filter with a smaller kernel, upscale.
 *   https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html
 * - Sigg & Hadwiger, GPU Gems 2, Chapter 20, "Fast Third-Order Texture Filtering":
 *   evaluate cubic B-splines with paired hardware-linear samples.
 *   https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering
 * - Bjorge, "Bandwidth-Efficient Rendering", SIGGRAPH 2015: multi-resolution
 *   filtering. Its mixed-resolution pipeline is not our same-target crossfade.
 *   https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
 */

/** @type {GPURenderPipeline} */
let pip_smol_gaussian = null;
let pip_smol_down = null;
let pip_smol_mix = null;
let pip_smol_mix_bilinear = null;
const smolUniformBuffers = [];
let smolUniformBufferIndex = 0;

function initSmolGaussian() {
    pip_smol_gaussian = createPipeline(SMOL_GAUSSIAN_SHADER);
    pip_smol_down = createPipeline(SMOL_DOWNSAMPLE_SHADER);
    // Two specializations of one reconstruction shader, not two blur algorithms.
    // The general variant can still use bilinear on either axis independently.
    pip_smol_mix = createPipeline(SMOL_MIX_SHADER);
    pip_smol_mix_bilinear = createPipeline(SMOL_MIX_SHADER, undefined, false,
        { ALL_BILINEAR: true });
}

const SMOL_GAUSSIAN_SHADER = FULLSCREEN_VERTEX_SHADER + `
struct Params {
    stepCenterCount: vec4f,
    taps: array<vec4f, 12>,
}
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var col = textureSampleLevel(tex, smp, in.uv, 0.0) * params.stepCenterCount.z;
    for (var i = 0u; i < u32(params.stepCenterCount.w); i++) {
        let tap = params.taps[i];
        let offset = params.stepCenterCount.xy * tap.x;
        col += (textureSampleLevel(tex, smp, in.uv - offset, 0.0)
              + textureSampleLevel(tex, smp, in.uv + offset, 0.0)) * tap.y;
    }
    return col;
}
`;

// Integrate each destination pixel's source footprint. Area weights preserve
// isolated highlights even when odd dimensions change sampling phase. Pairing
// weights into linear samples is exact apart from hardware filtering precision.
const SMOL_DOWNSAMPLE_SHADER = FULLSCREEN_VERTEX_SHADER + `
// xy is the logical size, zw is the one-texel padding on each axis.
struct Params { source: vec4f, destination: vec4f, }
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
// The area filter covers <=3 texels per axis. Pair its first two positive
// weights into one linear lookup; the remaining texel is the second group.
// Return two normalized coordinates and the second group's normalized weight.
fn areaAxis(lo: f32, hi: f32, physicalSize: f32) -> vec3f {
    let first = floor(lo);
    let w0 = min(hi, first + 1.0) - lo;
    let w1 = max(0.0, min(hi, first + 2.0) - (first + 1.0));
    let w2 = max(0.0, hi - (first + 2.0));
    return vec3f((first + 0.5 + w1 / (w0 + w1)) / physicalSize,
                 (first + 2.5) / physicalSize, w2 / (hi - lo));
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let physicalSize = vec2f(textureDimensions(tex));
    let size = params.source.xy;
    let outputSize = params.destination.xy;
    let scale = size / outputSize;
    let pixel = floor(in.position.xy) - params.destination.zw;
    var lo = pixel * scale + params.source.zw;
    var hi = min((pixel + 1.0) * scale, size) + params.source.zw;
    // Copy the source's outermost row/column into the border. On later
    // reductions these are the preserved edges, not averaged interior texels.
    // The other axis still integrates its footprint, including at corners.
    for (var axis = 0u; axis < 2u; axis++) {
        if (pixel[axis] < 0.0) {
            lo[axis] = 0.0;
            hi[axis] = 1.0;
        } else if (pixel[axis] >= outputSize[axis]) {
            lo[axis] = physicalSize[axis] - 1.0;
            hi[axis] = physicalSize[axis];
        }
    }
    // Exact halves, unchanged axes, and their borders need one bilinear read.
    if (all((size == outputSize) | (size == 2.0 * outputSize))) {
        return textureSampleLevel(tex, smp, (lo + hi) * 0.5 / physicalSize, 0.0);
    }
    let x = areaAxis(lo.x, hi.x, physicalSize.x);
    let y = areaAxis(lo.y, hi.y, physicalSize.y);
    // These decisions are uniform over the draw. Exact/unchanged axes need
    // only the first group, including their preserved border pixels.
    let secondX = size.x != outputSize.x && size.x != 2.0 * outputSize.x;
    let secondY = size.y != outputSize.y && size.y != 2.0 * outputSize.y;
    var a = textureSampleLevel(tex, smp, vec2f(x.x, y.x), 0.0);
    if (secondX) {
        a = mix(a, textureSampleLevel(tex, smp, vec2f(x.y, y.x), 0.0), x.z);
    }
    if (secondY) {
        var b = textureSampleLevel(tex, smp, vec2f(x.x, y.y), 0.0);
        if (secondX) {
            b = mix(b, textureSampleLevel(tex, smp, vec2f(x.y, y.y), 0.0), x.z);
        }
        a = mix(a, b, y.z);
    }
    return a;
}
`;

// Reconstruct and blend all RGBA channels, without using alpha as a weight.
const SMOL_MIX_SHADER = FULLSCREEN_VERTEX_SHADER + `
override ALL_BILINEAR: bool = false;
struct Params {
    ratio: vec2f, outputSize: vec2f,
    base: vec4f, neighbor: vec4f, diagonal: vec4f,
}
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var texX: texture_2d<f32>;
@group(0) @binding(4) var texY: texture_2d<f32>;
// Bilinear through 2× enlargement on each axis; positive cubic B-spline
// beyond that to avoid coarse-grid facets without ringing. This uses one,
// two, or four bilinear samples depending on how many axes need cubic.
// Pair adjacent positive B-spline weights: a sample at w1/(w0+w1) between
// two texel centers, multiplied by w0+w1, equals their weighted sum. Pairing
// four taps this way needs two linear reads per cubic axis (four in 2D).
// Each axis returns (first position, second position, second group weight).
fn cubicAxis(uv: f32, size: f32, outputSize: f32) -> vec3f {
    if (2.0 * size >= outputSize) { return vec3f(uv, uv, 0.0); }
    let p = uv * size - 0.5;
    let base = floor(p);
    let f = p - base;
    let w0 = (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0;
    let w1 = (3.0 * f * f * f - 6.0 * f * f + 4.0) / 6.0;
    let w2 = (-3.0 * f * f * f + 3.0 * f * f + 3.0 * f + 1.0) / 6.0;
    let w3 = f * f * f / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    return vec3f((base - 0.5 + w1 / g0) / size,
                 (base + 1.5 + w3 / g1) / size, g1);
}
// Coordinates and filter selection use the logical image, excluding padding.
fn paddedUV(uv: vec2f, imageBounds: vec4f) -> vec2f {
    return (uv * imageBounds.xy + imageBounds.zw) / (imageBounds.xy + 2.0 * imageBounds.zw);
}
fn reconstruct(image: texture_2d<f32>, uv: vec2f, imageBounds: vec4f) -> vec4f {
    // Pipeline specialization removes the cubic path for all-bilinear draws.
    if (ALL_BILINEAR) { return textureSampleLevel(image, smp, paddedUV(uv, imageBounds), 0.0); }
    let size = imageBounds.xy;
    let x = cubicAxis(uv.x, size.x, params.outputSize.x);
    let y = cubicAxis(uv.y, size.y, params.outputSize.y);
    var a = textureSampleLevel(image, smp, paddedUV(vec2f(x.x, y.x), imageBounds), 0.0);
    if (2.0 * size.x < params.outputSize.x) {
        a = mix(a, textureSampleLevel(image, smp, paddedUV(vec2f(x.y, y.x), imageBounds), 0.0), x.z);
    }
    if (2.0 * size.y < params.outputSize.y) {
        var b = textureSampleLevel(image, smp, paddedUV(vec2f(x.x, y.y), imageBounds), 0.0);
        if (2.0 * size.x < params.outputSize.x) {
            b = mix(b, textureSampleLevel(image, smp, paddedUV(vec2f(x.y, y.y), imageBounds), 0.0), x.z);
        }
        a = mix(a, b, y.z);
    }
    return a;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Triangle weights: base, dominant-axis neighbor, diagonal neighbor.
    var color = reconstruct(tex, in.uv, params.base) * (1.0 - params.ratio.x - params.ratio.y);
    if (params.ratio.x > 0.0) {
        color += reconstruct(texX, in.uv, params.neighbor) * params.ratio.x;
    }
    if (params.ratio.y > 0.0) {
        color += reconstruct(texY, in.uv, params.diagonal) * params.ratio.y;
    }
    return color;
}
`;

// ================ Smol Gaussian blur

function smolPass(encoder, pipeline, input, output, data, other = null) {
    const index = smolUniformBufferIndex++;
    if (!smolUniformBuffers[index]) {
        smolUniformBuffers[index] = gpu_device.createBuffer({
            size: 208, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    const buffer = smolUniformBuffers[index];
    gpu_device.queue.writeBuffer(buffer, 0, data);
    const entries = [
        { binding: 1, resource: input.createView() },
        { binding: 2, resource: { buffer } },
    ];
    entries.push({ binding: 0, resource: gpu_sampler_linear });
    if (other) other.forEach((tex, i) => entries.push({ binding: 3 + i, resource: tex.createView() }));
    const pass = encoder.beginRenderPass({colorAttachments: [{
        view: output.createView(), loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
    }]});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, gpu_device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries,
    }));
    pass.draw(3);
    pass.end();
}

function smolAxisPlan(radius, size) {
    const sigma = Math.max(0, radius) / 3;
    let level = 0;
    // Ceil-sized levels support odd dimensions and stop at a single texel.
    // Use the actual size ratio, not 2**level: e.g. 5 -> 3 is a 5/3 reduction.
    // At working sigma 5 start the crossfade, and at 6 advance the base level.
    // Both endpoints still use the original target sigma in endpoint().
    const maxLevel = Math.ceil(Math.log2(size));
    const scale = level => size / Math.ceil(size / 2 ** level);
    while (level < maxLevel && sigma >= 6 * scale(level)) level++;
    const t = Math.min(1, Math.max(0, sigma / scale(level) - 5));
    return { sigma, level, blend: level < maxLevel ? t * t * (3 - 2 * t) : 0 };
}

function smolGaussianAxis(encoder, input, sigma, horizontal, destination = null) {
    if (sigma < 0.01 || (horizontal ? input.width : input.height) === 1) return input;
    const data = new Float32Array(52);
    data[0] = horizontal ? 1 / input.width : 0;
    data[1] = horizontal ? 0 : 1 / input.height;
    // Smoothly taper the last sigma of the support. Changing tap count then
    // introduces zero-weight taps, avoiding small jumps during animation.
    const weight = i => {
        const x = i / sigma;
        const t = Math.max(0, Math.min(1, 4 - x));
        return Math.exp(-0.5 * x * x) * t * t * (3 - 2 * t);
    };
    // Twelve paired taps cover 24 texels on each side, enough for 4*sigma
    // while the normal level plan keeps working sigma below 6. If reduction
    // stops at one logical texel, the cap still bounds the cost but truncates
    // wider kernels; this extreme case remains an approximation.
    const count = Math.min(12, Math.ceil(4 * sigma / 2));
    let sum = 1;
    for (let j = 0; j < count; j++) {
        const i = 2 * j + 1;
        // One linear sample at i + b/(a+b), weighted by a+b, replaces the
        // two neighboring taps exactly (apart from sampler/float precision).
        const a = weight(i), b = weight(i + 1), w = a + b;
        data[4 + j * 4] = w > 0 ? i + b / w : i;
        data[5 + j * 4] = w;
        sum += 2 * w;
    }
    data[2] = 1 / sum;
    data[3] = count;
    for (let j = 0; j < count; j++) data[5 + j * 4] /= sum;
    const output = destination ?? getCachedTexture(input.width, input.height);
    smolPass(encoder, pip_smol_gaussian, input, output, data);
    return output;
}

// Reconstruction variance in output pixels, matching the shader's per-axis
// filter choice. At exact 2×, bilinear phases are 1/4 and 3/4 (variance 3/16
// in source texels); fractional odd-size scales use the phase average 1/6.
function reconstructionVariance(scale) {
    if (scale === 1) return 0;
    return scale * scale * (scale > 2 ? 1 / 3 : scale === 2 ? 3 / 16 : 1 / 6);
}

function blurSmolGaussian(encoder, input, output) {
    const x = smolAxisPlan(blur_params.radius_x, input.width);
    const y = smolAxisPlan(blur_params.radius_y, input.height);
    // Keep the selected levels, but skip neighboring endpoints and switch
    // levels abruptly at the existing reduction thresholds.
    if (blur_params.disable_smol_blending) x.blend = y.blend = 0;
    const levels = new Map();
    levels.set('0,0', { texture: input, width: input.width, height: input.height,
        padX: 0, padY: 0, vx: 0, vy: 0 });
    // Share a canonical reduction tree between at most three endpoints.
    function down(lx, ly) {
        const key = `${lx},${ly}`;
        if (levels.has(key)) return levels.get(key);
        // First reduce both axes together to their common level, then the
        // remaining axis. Neighboring rectangular endpoints thus reuse the
        // expensive large-image prefix instead of starting shifted pyramids.
        // This route depends only on the endpoint, not the blend weights: odd
        // dimensions must follow the same resampling path across transitions.
        const prev = lx > ly ? down(lx - 1, ly)
            : ly > lx ? down(lx, ly - 1) : down(lx - 1, ly - 1);
        const src = prev.texture;
        const width = Math.ceil(input.width / 2 ** lx);
        const height = Math.ceil(input.height / 2 ** ly);
        // Only reduced axes need padding; zero-radius axes keep exact texels.
        const padX = lx > 0 ? 1 : 0;
        const padY = ly > 0 ? 1 : 0;
        const texture = getCachedTexture(width + 2 * padX, height + 2 * padY);
        smolPass(encoder, pip_smol_down, src, texture,
            new Float32Array([prev.width, prev.height, prev.padX, prev.padY,
                width, height, padX, padY]));
        // Box reduction variance in original texels; approximate for odd sizes.
        const entry = { texture, width, height, padX, padY,
            vx: ((input.width / width) ** 2 - 1) / 12,
            vy: ((input.height / height) ** 2 - 1) / 12,
        };
        levels.set(key, entry);
        intermediateTextures.push({ texture, name: `Reduced ${width}×${height}` });
        return entry;
    }
    function endpoint(lx, ly, destination = null) {
        const entry = down(lx, ly);
        const sx = input.width / entry.width;
        const sy = input.height / entry.height;
        // Variances add under convolution. In original-image pixel units:
        // targetSigma^2 ~= reductionVariance + (scale * residualSigma)^2
        //                  + reconstructionVariance.
        // Solve per axis, clamp at zero, then convert sigma to reduced texels.
        // Resampling is phase-dependent, so this is a width estimate rather
        // than an exact description of the resulting kernel or its borders.
        const sigmaX = Math.sqrt(Math.max(0, x.sigma ** 2 - entry.vx - reconstructionVariance(sx))) / sx;
        const sigmaY = Math.sqrt(Math.max(0, y.sigma ** 2 - entry.vy - reconstructionVariance(sy))) / sy;
        const filterY = sigmaY >= 0.01 && entry.texture.height > 1;
        let tex = smolGaussianAxis(encoder, entry.texture, sigmaX, true,
            filterY ? null : destination);
        tex = smolGaussianAxis(encoder, tex, sigmaY, false, destination);
        intermediateTextures.push({ texture: tex, name: `Gaussian ${tex.width}×${tex.height}` });
        return { ...entry, texture: tex };
    }
    // Reconstruct each endpoint directly on the output grid. An intermediate
    // resize here would add blur that disappears at the level boundary.
    // With no resizing or blending, write the final Gaussian axis directly
    // to the output instead of running a full-resolution reconstruction pass.
    const direct = x.level === 0 && y.level === 0 && x.blend === 0 && y.blend === 0;
    const a = endpoint(x.level, y.level, direct ? output : null);
    if (direct) {
        // Both axes can be bypassed for zero/tiny radii or one-pixel images.
        if (a.texture !== output) encoder.copyTextureToTexture(
            { texture: a.texture }, { texture: output }, [output.width, output.height]);
        return;
    }
    const middleWeight = Math.abs(x.blend - y.blend);
    const diagonalWeight = Math.min(x.blend, y.blend);
    // Split the blend square along (0,0)--(1,1). On that diagonal only
    // matching-resolution endpoints contribute; elsewhere use one neighbor.
    // For fractions tx,ty the weights are 1-max(tx,ty), abs(tx-ty), min(tx,ty).
    // They are nonnegative and sum to one. Compared with four-corner bilinear
    // interpolation this saves one endpoint, with a possible slope change at
    // the diagonal; shared edge weights keep the result continuous.
    const b = middleWeight > 0
        ? endpoint(x.level + (x.blend > y.blend ? 1 : 0),
                   y.level + (y.blend > x.blend ? 1 : 0)) : a;
    const c = diagonalWeight > 0 ? endpoint(x.level + 1, y.level + 1) : a;
    // Inactive neighbors alias a. Check every sampled endpoint, including
    // coarser neighbors during transitions, before choosing the simple shader.
    const allBilinear = [a, b, c].every(tex =>
        2 * tex.width >= output.width && 2 * tex.height >= output.height);
    const bounds = entry => [entry.width, entry.height, entry.padX, entry.padY];
    smolPass(encoder, allBilinear ? pip_smol_mix_bilinear : pip_smol_mix, a.texture, output,
        new Float32Array([middleWeight, diagonalWeight, output.width, output.height,
            ...bounds(a), ...bounds(b), ...bounds(c)]), [b.texture, c.texture]);
}


/* Extended Dual Kawase, using a five-sample downsample filter and an eight-sample
 * upsample filter. The original uses equal horizontal/vertical blur amounts,
 * and only supports a discrete "number of blur pyramid levels" control.
 *
 * For arbitrary blur sizes, blend between neighboring discrete blur levels,
 * similar to https://github.com/FiniteSingularity/obs-composite-blur.
 * The fractional position between steps is remapped with t * (2 + t) / 3,
 * which makes it feel a bit nicer than just a linear blend.
 * For independent X/Y radii, stop further reductions along an axis when its
 * smaller blur radius is reached. Between levels, this can mean blending three
 * different blurred results; equal radii only need two.
 *
 * Endpoints share their downsample prefixes, are blended on a common grid,
 * then use a shared upsample suffix. Since upsampling is linear, this avoids
 * separate full-resolution reconstructions. Sampling paths and grids must match
 * to preserve the result.
 *
 * Radius/3 is an approximate visual mapping, not a measured Gaussian sigma.
 * The result kinda works, but smoothly animated radius does not "feel" smooth
 * on HDR highlights, due to blending between discrete blur levels.
 *
 * Original filters: Marius Bjorge, "Bandwidth-Efficient Rendering" (SIGGRAPH 2015):
 * https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
 * Also explained at https://blog.frost.kiwi/dual-kawase/#dual-kawase-blur.
 */


/** @type {GPURenderPipeline} */
let pip_dk_down = null;
/** @type {GPURenderPipeline} */
let pip_dk_up = null;
/** @type {GPURenderPipeline} */
let pip_dk_mix = null;

// Reusable resources for Kawase blurs
/** @type {GPUBuffer[]} */
let kawaseUniformBuffers = [];
let kawaseUniformBufferIndex = 0;

function initDualKawase() {
    pip_dk_down = createPipeline(DK_DOWNSAMPLE_SHADER);
    pip_dk_up = createPipeline(DK_UPSAMPLE_SHADER);
    pip_dk_mix = createPipeline(DK_MIX_SHADER, undefined, true);
}

const DK_DOWNSAMPLE_SHADER = FULLSCREEN_VERTEX_SHADER + `
struct Params {
    uvStep: vec2f,
}
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var col = vec4f(0.0);
    col += textureSample(tex, smp, in.uv) * 4.0;
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f(-0.5, -0.5));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f(-0.5,  0.5));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 0.5, -0.5));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 0.5,  0.5));
    return col / 8.0;
}
`;

const DK_UPSAMPLE_SHADER = FULLSCREEN_VERTEX_SHADER + `
struct Params {
    uvStep: vec2f,
}
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var col = vec4f(0.0);
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 0.0, -1.0));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 0.0,  1.0));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f(-1.0,  0.0));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 1.0,  0.0));
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f(-0.5, -0.5)) * 2.0;
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f(-0.5,  0.5)) * 2.0;
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 0.5, -0.5)) * 2.0;
    col += textureSample(tex, smp, in.uv + params.uvStep * vec2f( 0.5,  0.5)) * 2.0;
    return col / 12.0;
}
`;

const DK_MIX_SHADER = FULLSCREEN_VERTEX_SHADER + `
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex, smp, in.uv);
}
`;

// ================ Dual Kawase blur

// Get a uniform buffer from pool (creates if needed)
function getKawaseUniformBuffer() {
    if (kawaseUniformBufferIndex >= kawaseUniformBuffers.length) {
        kawaseUniformBuffers.push(gpu_device.createBuffer({
            size: 8,  // uvStep (vec2f)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }));
    }
    return kawaseUniformBuffers[kawaseUniformBufferIndex++];
}

// Reset uniform buffer pool index at start of frame
function resetKawaseUniformBufferPool() {
    kawaseUniformBufferIndex = 0;
}

function kawaseDownsample(commandEncoder, input, fullSizeX, fullSizeY, divisorX, divisorY, ratioX, ratioY) {
    const sizeX = Math.max(Math.floor(fullSizeX / Math.max(divisorX,1)), 1);
    const sizeY = Math.max(Math.floor(fullSizeY / Math.max(divisorY,1)), 1);
    const output = getCachedTexture(sizeX, sizeY);

    // Downsample offsets use destination texels; for an exact half-size step,
    // the shader's +/-0.5 offset is one source texel. Odd sizes use the actual
    // destination dimensions. ratioX/Y are axis masks (0 or 1), not blend weights.
    const stepX = ratioX / sizeX;
    const stepY = ratioY / sizeY;

    const paramsData = new Float32Array([stepX, stepY]);
    const paramsBuffer = getKawaseUniformBuffer();
    gpu_device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const bindGroup = gpu_device.createBindGroup({
        layout: pip_dk_down.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: gpu_sampler_linear },
            { binding: 1, resource: input.createView() },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });

    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: output.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
    });

    renderPass.setPipeline(pip_dk_down);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.end();

    return output;
}

function kawaseUpsample(commandEncoder, input, fullSizeX, fullSizeY, divisorX, divisorY, ratioX, ratioY) {
    const sizeX = Math.max(Math.floor(fullSizeX / Math.max(divisorX,1)), 1);
    const sizeY = Math.max(Math.floor(fullSizeY / Math.max(divisorY,1)), 1);
    const output = getCachedTexture(sizeX, sizeY);

    // Upsample offsets use source texels. An axis that is inactive in this
    // pass has zero offset; its resolution is controlled separately by divisor.
    // The ratioX/Y axis masks are unrelated to the interpolation fraction.
    const stepX = ratioX / input.width;
    const stepY = ratioY / input.height;

    const paramsData = new Float32Array([stepX, stepY]);
    const paramsBuffer = getKawaseUniformBuffer();
    gpu_device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const bindGroup = gpu_device.createBindGroup({
        layout: pip_dk_up.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: gpu_sampler_linear },
            { binding: 1, resource: input.createView() },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });

    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: output.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
    });

    renderPass.setPipeline(pip_dk_up);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.end();

    return output;
}

function kawaseMix(commandEncoder, input, output, ratio) {
    const bindGroup = gpu_device.createBindGroup({
        layout: pip_dk_mix.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: gpu_sampler_point },
            { binding: 1, resource: input.createView() },
        ],
    });

    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: output.createView(),
            loadOp: 'load',  // Important: load existing content for blending
            storeOp: 'store',
        }],
    });

    // Use a constant weight so interpolation preserves all four channels.
    renderPass.setBlendConstant([ratio, ratio, ratio, ratio]);
    renderPass.setPipeline(pip_dk_mix);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.end();
}

function kawaseCopy(commandEncoder, input, output) {
    commandEncoder.copyTextureToTexture(
        { texture: input },
        { texture: output },
        [input.width, input.height]
    );
}

// Discrete radii are 0, 6, 12, 24, ... (pyramid divisors 0, 2, 4, 8, ...).
function kawaseRadiusInterval(radius) {
    const steps = Math.max(radius, 0) / 3.0;
    let lower = 0;
    let upper = 2;
    while (upper <= steps) {
        lower = upper;
        upper *= 2;
    }
    const t = (steps - lower) / (upper - lower);
    // Blending kernels interpolates variance. Remap the weight to make blur
    // width grow more evenly between doubling steps. If endpoint widths are
    // r and 2r and the desired width is r*(1+t), the variance-matching weight
    // is ((1+t)^2 - 1)/(4 - 1) = t*(2+t)/3. Real Kawase kernels only roughly
    // follow this model. We also use the remap for 0 -> 6 as a heuristic;
    // that first interval does not satisfy the doubling-width derivation.
    return { lower, upper, fraction: t * (2.0 + t) / 3.0 };
}

function blurDualKawase(commandEncoder, input, output) {
    const x = kawaseRadiusInterval(blur_params.radius_x);
    const y = kawaseRadiusInterval(blur_params.radius_y);
    const fullSizeX = cur_image_size.width;
    const fullSizeY = cur_image_size.height;

    // Triangulate the radius cell along its lower/lower -> upper/upper diagonal.
    // Equal radii use just the two isotropic endpoints; elsewhere at most three
    // discrete kernels contribute. Adjacent triangles/cells share their edges.
    // With remapped fractions tx,ty, weights are 1-max(tx,ty), abs(tx-ty),
    // min(tx,ty). They sum to one; the middle endpoint advances whichever
    // axis has the larger fraction. On the diagonal its weight is zero.
    // This saves the fourth corner of bilinear interpolation, at the cost
    // of a possible slope change along the triangle boundary.
    const corners = [
        { x: x.lower, y: y.lower, weight: 1.0 - Math.max(x.fraction, y.fraction) },
        x.fraction >= y.fraction
            ? { x: x.upper, y: y.lower, weight: x.fraction - y.fraction }
            : { x: x.lower, y: y.upper, weight: y.fraction - x.fraction },
        { x: x.upper, y: y.upper, weight: Math.min(x.fraction, y.fraction) },
    ].filter(corner => corner.weight > 0.0);

    // Every endpoint shares this isotropic downsample prefix and upsample suffix.
    // Blend BEFORE the shared upsampling: the filters are linear, so there is no
    // need to render several full-resolution blurs. For X=Y this is precisely one
    // pyramid plus an extra down/up pair and a mix at its smallest shared level.
    const commonLevel = Math.min(x.lower, y.lower);
    let curr = input;
    for (let i = 2; i <= commonLevel; i *= 2) {
        curr = kawaseDownsample(commandEncoder, curr, fullSizeX, fullSizeY, i, i, 1.0, 1.0);
        intermediateTextures.push({ texture: curr, name: `Down (1/${i}): ${curr.width}×${curr.height}` });
    }
    const common = curr;
    const downsampleCache = new Map();

    function renderCorner(corner) {
        let texture = common;
        const minLevel = Math.min(corner.x, corner.y);
        const maxLevel = Math.max(corner.x, corner.y);
        let path = '';
        for (let i = Math.max(commonLevel, 1) * 2; i <= maxLevel; i *= 2) {
            const doX = i <= corner.x;
            const doY = i <= corner.y;
            const divisorX = doX ? i : minLevel;
            const divisorY = doY ? i : minLevel;
            // Also share any downsample prefix within the divergent branches.
            path += `/${divisorX},${divisorY},${doX},${doY}`;
            let down = downsampleCache.get(path);
            if (!down) {
                down = kawaseDownsample(commandEncoder, texture, fullSizeX, fullSizeY,
                    divisorX, divisorY, doX ? 1.0 : 0.0, doY ? 1.0 : 0.0);
                downsampleCache.set(path, down);
                intermediateTextures.push({ texture: down, name: `Corner (${corner.x},${corner.y}) down: ${down.width}×${down.height}` });
            }
            texture = down;
        }
        // Keep the existing discrete pass sequence, including its inactive-axis
        // divisors and offsets. Stop where the common upsample suffix begins.
        for (let i = Math.floor(maxLevel / 2); i >= Math.max(commonLevel, 1); i = Math.floor(i / 2)) {
            texture = kawaseUpsample(commandEncoder, texture, fullSizeX, fullSizeY,
                i <= corner.x ? i : minLevel, i <= corner.y ? i : minLevel,
                i <= corner.x ? 1.0 : 0.0, i <= corner.y ? 1.0 : 0.0);
            intermediateTextures.push({ texture, name: `Corner (${corner.x},${corner.y}) up: ${texture.width}×${texture.height}` });
        }
        return texture;
    }

    const textures = corners.map(renderCorner);
    // The last corner owns a filtered texture whenever blending is needed;
    // never blend into the source or the saved common pyramid texture.
    curr = textures[textures.length - 1];
    // Maintain a normalized weighted average of the accumulated corners.
    // Each mix uses incomingWeight / newTotalWeight, not the raw corner
    // weight, so two in-place blends reproduce the three-corner weighted sum.
    let weight = corners[corners.length - 1].weight;
    for (let i = corners.length - 2; i >= 0; --i) {
        weight += corners[i].weight;
        kawaseMix(commandEncoder, textures[i], curr, corners[i].weight / weight);
    }
    if (corners.length > 1) {
        intermediateTextures.push({ texture: curr, name: `Interpolated (${blur_params.radius_x},${blur_params.radius_y}): ${curr.width}×${curr.height}` });
    }

    for (let i = Math.floor(commonLevel / 2); i >= 1; i = Math.floor(i / 2)) {
        curr = kawaseUpsample(commandEncoder, curr, fullSizeX, fullSizeY, i, i, 1.0, 1.0);
        if (i > 1) {
            intermediateTextures.push({ texture: curr, name: `Up (1/${i}): ${curr.width}×${curr.height}` });
        }
    }
    kawaseCopy(commandEncoder, curr, output);
}


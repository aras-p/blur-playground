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

    // Note: "step" is based on *output* size, which is 2x smaller than input
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

    // Note: "step" is based on *input* size, which is 2x smaller than output
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
    // width grow more evenly between doubling steps.
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


// Box, tent, and Gaussian convolution; also used by Fast Gaussian at small radii.

/** @type {GPURenderPipeline} */
let pip_separable = null;

function initSeparableBlur() {
    pip_separable = createPipeline(SEPARABLE_BLUR_SHADER);
}

const SEPARABLE_BLUR_SHADER = FULLSCREEN_VERTEX_SHADER + `
struct BlurParams {
    uvStep: vec2f,
    kernelWidth: i32,
}
@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var texKernel: texture_1d<f32>;
@group(0) @binding(3) var<uniform> params: BlurParams;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var col = textureSample(tex, smp, in.uv) * textureLoad(texKernel, 0, 0).r;
    for (var i: i32 = 1; i < params.kernelWidth; i++) {
        let w = textureLoad(texKernel, i, 0).r;
        let offset = params.uvStep * f32(i);
        col += textureSample(tex, smp, in.uv - offset) * w;
        col += textureSample(tex, smp, in.uv + offset) * w;
    }
    return col;
}
`;

// ================ Blur kernel utilities

function separableKernelValue(mode, x) {
    x = Math.abs(x);
    switch (mode) {
        case BlurMode.BOX:
            return x > 1.0 ? 0.0 : 1.0;
        case BlurMode.TENT:
            return x > 1.0 ? 0.0 : 1.0 - x;
        case BlurMode.GAUSSIAN: {
            const scale = 1.6;
            const twoScale2 = 2.0 * scale * scale;
            x *= 3.0 * scale;
            return (1.0 / Math.sqrt(Math.PI * twoScale2)) * Math.exp(-x * x / twoScale2);
        }
        default:
            return 0.0;
    }
}

function calcSeparableWeights(mode, radius) {
    // Kernel size is radius+1, but since it is symmetric we only store one half
    const size = Math.ceil(radius) + 1;
    const weights = new Float32Array(size);

    let sum = 0.0;

    // Center weight
    const centerWeight = separableKernelValue(mode, 0.0);
    weights[0] = centerWeight;
    sum += centerWeight;

    // Other weights in the positive direction
    // Add double to the sum to account for negative direction
    const scale = radius > 0.0 ? 1.0 / radius : 0.0;
    for (let i = 1; i < size; i++) {
        const weight = separableKernelValue(mode, i * scale);
        weights[i] = weight;
        sum += weight * 2.0;
    }

    // Normalize the weights
    for (let i = 0; i < size; i++) {
        weights[i] /= sum;
    }

    // Create 1D texture for kernel weights
    const kernelTexture = gpu_device.createTexture({
        size: [size],
        format: 'r32float',
        dimension: '1d',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    gpu_device.queue.writeTexture(
        { texture: kernelTexture },
        weights,
        { bytesPerRow: size * 4 },
        { width: size }
    );

    return { texture: kernelTexture, width: size };
}

function separablePass(commandEncoder, input, output, mode, horizontal, radius) {
    const width = cur_image_size.width;
    const height = cur_image_size.height;

    // Calculate kernel weights
    const kernel = calcSeparableWeights(mode, radius);

    // Create uniform buffer for blur params
    const paramsData = new ArrayBuffer(16); // vec2f (8) + i32 (4) + padding (4)
    const paramsView = new DataView(paramsData);
    paramsView.setFloat32(0, horizontal ? 1.0 / width : 0.0, true);
    paramsView.setFloat32(4, horizontal ? 0.0 : 1.0 / height, true);
    paramsView.setInt32(8, kernel.width, true);

    const paramsBuffer = gpu_device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    gpu_device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Create bind group for this pass
    const blurBindGroup = gpu_device.createBindGroup({
        layout: pip_separable.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: gpu_sampler_linear },
            { binding: 1, resource: input.createView() },
            { binding: 2, resource: kernel.texture.createView() },
            { binding: 3, resource: { buffer: paramsBuffer } },
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

    renderPass.setPipeline(pip_separable);
    renderPass.setBindGroup(0, blurBindGroup);
    renderPass.draw(3);
    renderPass.end();

    // Resources (kernel texture, params buffer) will be garbage collected after GPU is done
}


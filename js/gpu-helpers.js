// Shared GPU shaders, resource creation, and texture pooling.

const FULLSCREEN_VERTEX_SHADER = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var o: VertexOutput;
    // full-screen triangle
    o.uv = vec2f(select(0.0, 2.0, (vid & 1u) != 0u), select(1.0, -1.0, (vid & 2u) != 0u));
    o.position = vec4f(o.uv.x * 2.0 - 1.0, -(o.uv.y * 2.0 - 1.0), 0.0, 1.0);
    return o;
}
`;

const blurTextureCache = new Map();

function createTexture2D(width, height, extraUsage = 0) {
    return gpu_device.createTexture({
        size: [width, height],
        format: 'rgba32float',
        usage: extraUsage | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
}

function createSampler(filter) {
    return gpu_device.createSampler({
        magFilter: filter,
        minFilter: filter,
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
    })
}

function createPipeline(codeString, rtFormat = 'rgba32float', blending = false, constants = {}) {
    const shader_mod = gpu_device.createShaderModule({
        code: codeString,
    });
    const target = { format: rtFormat };
    if (blending) {
        target.blend = {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add', },
            alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add', },
        };
    }
    return gpu_device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shader_mod, entryPoint: 'vs_main', },
        fragment: { module: shader_mod, entryPoint: 'fs_main', targets: [target], constants, },
        primitive: { topology: 'triangle-list', },
    })
}

// Get a cached texture with matching dimensions and usage.
function getCachedTexture(width, height, extraUsage = 0) {
    const key = `${width},${height},${extraUsage}`;
    let textures = blurTextureCache.get(key);
    if (!textures) {
        textures = [];
        blurTextureCache.set(key, textures);
    }
    // Find an unused texture or create new one
    for (const entry of textures) {
        if (!entry.inUse) {
            entry.inUse = true;
            return entry.texture;
        }
    }
    // Create new texture
    const texture = createTexture2D(width, height, extraUsage);
    textures.push({ texture, inUse: true });
    return texture;
}

// Mark all cached textures as unused (call at start of frame)
function resetTextureCache() {
    for (const textures of blurTextureCache.values()) {
        for (const entry of textures) {
            entry.inUse = false;
        }
    }
}

// Clear texture cache (call when image size changes)
function clearTextureCache() {
    for (const textures of blurTextureCache.values()) {
        for (const entry of textures) {
            entry.texture.destroy();
        }
    }
    blurTextureCache.clear();
}


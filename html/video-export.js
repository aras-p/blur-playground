// Both live animation and export use this duration and the same radius function.
const BLUR_ANIMATION_DURATION = 6000;

async function exportBlurVideo({ canvas, renderFrame, onProgress }) {
    const fps = 60;
    const frameCount = BLUR_ANIMATION_DURATION / 1000 * fps;
    // AVC needs even dimensions. Pad odd images by one pixel, without resizing.
    const capture = document.createElement('canvas');
    capture.width = Math.ceil(canvas.width / 2) * 2;
    capture.height = Math.ceil(canvas.height / 2) * 2;
    const context = capture.getContext('2d', { alpha: false });
    let config;
    // Baseline H.264 excludes B-frames, so decode order equals presentation order.
    // Try levels 4.2 (1080p60) and 5.2 (4K60), checking the actual dimensions.
    configurations: for (const hardwareAcceleration of ['prefer-hardware', 'no-preference']) {
        for (const codec of ['avc1.42002a', 'avc1.420034']) {
            const candidate = {
                codec, width: capture.width, height: capture.height,
                bitrate: 12_000_000, framerate: fps,
                hardwareAcceleration, latencyMode: 'quality', avc: { format: 'avc' },
            };
            if ((await VideoEncoder.isConfigSupported(candidate)).supported) {
                config = candidate;
                break configurations;
            }
        }
    }
    if (!config) {
        throw new Error(`H.264 encoding is unavailable at ${capture.width}×${capture.height}. Try a smaller image or another browser.`);
    }

    const samples = [];
    let description;
    let encodingError;
    const encoder = new VideoEncoder({
        output(chunk, metadata) {
            const data = new Uint8Array(chunk.byteLength);
            chunk.copyTo(data);
            // Fixed-rate MP4 tables below assume no dropped or reordered frames.
            if (chunk.timestamp !== Math.round(samples.length * 1_000_000 / fps)) {
                encodingError = new Error('The encoder dropped or reordered a video frame.');
            }
            samples.push({ data, key: chunk.type === 'key' });
            if (metadata.decoderConfig?.description) {
                const next = new Uint8Array(metadata.decoderConfig.description);
                if (description && (next.length !== description.length || next.some((v, i) => v !== description[i]))) {
                    encodingError = new Error('The encoder changed its H.264 configuration during export.');
                }
                description = next.slice();
            }
        },
        error(error) { encodingError = error; },
    });
    try {
        encoder.configure(config);
        for (let i = 0; i < frameCount; i++) {
            if (encodingError) throw encodingError;
            await renderFrame(i * 1000 / fps, () => {
                context.drawImage(canvas, 0, 0);
                const timestamp = Math.round(i * 1_000_000 / fps);
                const frame = new VideoFrame(capture, {
                    timestamp, duration: Math.round((i + 1) * 1_000_000 / fps) - timestamp,
                });
                try {
                    encoder.encode(frame, { keyFrame: i % (fps * 2) === 0 });
                } finally {
                    frame.close();
                }
            });
            // Bound queued frames and GPU work, and yield so progress can paint.
            if (encoder.encodeQueueSize >= 8) await encoder.flush();
            if ((i + 1) % 12 === 0 || i === frameCount - 1) {
                onProgress(i + 1, frameCount);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        await encoder.flush();
        if (encodingError) throw encodingError;
        if (samples.length !== frameCount || !description || !samples[0].key) {
            throw new Error('The encoder did not produce a complete H.264 video.');
        }
        return makeBlurMP4(samples, description, capture.width, capture.height, fps);
    } finally {
        if (encoder.state !== 'closed') encoder.close();
    }
}

// Minimal ISO BMFF writer for one constant-frame-rate AVC video track, no audio,
// no B-frames, and files below 4 GiB. VideoEncoder supplies length-prefixed NAL
// units and the AVCDecoderConfigurationRecord used verbatim in the avcC box.
function makeBlurMP4(samples, avcConfig, width, height, fps) {
    const bytes = (...values) => new Uint8Array(values);
    const zeros = size => new Uint8Array(size);
    const text = value => new TextEncoder().encode(value);
    const u16 = value => bytes(value >>> 8, value);
    const u32 = (...values) => {
        const result = new Uint8Array(values.length * 4);
        const view = new DataView(result.buffer);
        values.forEach((value, i) => view.setUint32(i * 4, value));
        return result;
    };
    const join = parts => {
        const result = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
        let offset = 0;
        for (const part of parts) { result.set(part, offset); offset += part.length; }
        return result;
    };
    const box = (type, ...parts) => join([u32(8 + parts.reduce((sum, part) => sum + part.length, 0)), text(type), ...parts]);
    const fullBox = (type, flags, ...parts) => box(type, u32(flags), ...parts);
    const matrix = u32(0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000);
    const count = samples.length;
    const dataSize = samples.reduce((sum, sample) => sum + sample.data.length, 0);
    const ftyp = box('ftyp', text('isom'), u32(512), text('isomiso2avc1mp41'));
    if (dataSize + ftyp.length + 8 >= 0x100000000) throw new Error('Video is too large for this MP4 exporter.');

    // All frames occupy one contiguous chunk, immediately after ftyp + mdat header.
    const stbl = box('stbl',
        fullBox('stsd', 0, u32(1), box('avc1',
            zeros(6), u16(1), zeros(16), u16(width), u16(height),
            u32(0x480000, 0x480000, 0), u16(1), zeros(32), u16(24), u16(0xffff),
            box('avcC', avcConfig))),
        fullBox('stts', 0, u32(1, count, 1)),
        fullBox('stsc', 0, u32(1, 1, count, 1)),
        fullBox('stsz', 0, u32(0, count), u32(...samples.map(sample => sample.data.length))),
        fullBox('stco', 0, u32(1, ftyp.length + 8)),
        fullBox('stss', 0, u32(samples.filter(sample => sample.key).length),
            u32(...samples.flatMap((sample, i) => sample.key ? [i + 1] : []))),
    );
    const moov = box('moov',
        fullBox('mvhd', 0, u32(0, 0, fps, count, 0x10000), u16(0x100), zeros(10), matrix, zeros(24), u32(2)),
        box('trak',
            fullBox('tkhd', 7, u32(0, 0, 1, 0, count), zeros(8), zeros(8), matrix, u32(width * 65536, height * 65536)),
            box('mdia',
                fullBox('mdhd', 0, u32(0, 0, fps, count), u16(0x55c4), u16(0)), // language: und
                fullBox('hdlr', 0, u32(0), text('vide'), zeros(12), text('Blur Video\0')),
                box('minf',
                    fullBox('vmhd', 1, zeros(8)),
                    box('dinf', fullBox('dref', 0, u32(1), fullBox('url ', 1))),
                    stbl))),
    );
    return new Blob([ftyp, u32(dataSize + 8), text('mdat'), ...samples.map(sample => sample.data), moov], { type: 'video/mp4' });
}

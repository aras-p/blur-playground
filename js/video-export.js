// Both live animation and export use this duration and the same radius function.
const BLUR_ANIMATION_DURATION = 8000;

async function exportBlurVideo({ canvas, renderFrame, onProgress }) {
    const fps = 60;
    const frameCount = BLUR_ANIMATION_DURATION / 1000 * fps;
    // Scale only the export; blur rendering stays at the source resolution.
    const scale = Math.min(1, 960 / Math.max(canvas.width, canvas.height));
    const width = Math.max(1, Math.round(canvas.width * scale));
    const height = Math.max(1, Math.round(canvas.height * scale));
    // AVC needs even dimensions. Pad odd sizes by one pixel after scaling.
    const capture = document.createElement('canvas');
    capture.width = Math.ceil(width / 2) * 2;
    capture.height = Math.ceil(height / 2) * 2;
    const context = capture.getContext('2d', { alpha: false });
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    let config;
    // AV1 Main, 8-bit, level 4.0 covers the export's maximum 960×960 at 60 fps.
    // Baseline H.264 fallback excludes B-frames, keeping presentation order.
    for (const codec of ['av01.0.08M.08', 'avc1.42002a', 'avc1.420034']) {
        const candidate = {
            codec, width: capture.width, height: capture.height,
            bitrate: 4_000_000, bitrateMode: 'variable', framerate: fps,
            latencyMode: 'quality',
            ...(codec.startsWith('avc1') ? { avc: { format: 'avc' } } : {}),
        };
        if ((await VideoEncoder.isConfigSupported(candidate)).supported) {
            config = candidate;
            break;
        }
    }
    if (!config) {
        throw new Error(`AV1 and H.264 encoding are unavailable at ${capture.width}×${capture.height}. Try a smaller image or another browser.`);
    }
    const isAV1 = config.codec.startsWith('av01');

    const samples = [];
    let description;
    let encodingError;
    let resumeQueue;
    const encoder = new VideoEncoder({
        output(chunk, metadata) {
            const data = new Uint8Array(chunk.byteLength);
            chunk.copyTo(data);
            // Fixed-rate MP4 tables below assume no dropped or reordered frames.
            if (chunk.timestamp !== Math.round(samples.length * 1_000_000 / fps)) {
                encodingError = new Error('The encoder dropped or reordered a video frame.');
            }
            samples.push({ data, key: chunk.type === 'key' });
            if (isAV1 && !description && chunk.type === 'key') {
                try { description = makeBlurAV1Config(data); }
                catch (error) { encodingError = error; }
            }
            if (!isAV1 && metadata.decoderConfig?.description) {
                const next = new Uint8Array(metadata.decoderConfig.description);
                if (description && (next.length !== description.length || next.some((v, i) => v !== description[i]))) {
                    encodingError = new Error('The encoder changed its H.264 configuration during export.');
                }
                description = next.slice();
            }
        },
        error(error) { encodingError = error; resumeQueue?.(); },
    });
    encoder.addEventListener('dequeue', () => resumeQueue?.());
    try {
        encoder.configure(config);
        for (let i = 0; i < frameCount; i++) {
            if (encodingError) throw encodingError;
            await renderFrame(i * 1000 / fps, () => {
                context.drawImage(canvas, 0, 0, width, height);
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
            while (encoder.encodeQueueSize >= 8 && !encodingError) {
                await new Promise(resolve => { resumeQueue = resolve; });
                resumeQueue = undefined;
            }
            if (encodingError) throw encodingError;
            if ((i + 1) % 12 === 0 || i === frameCount - 1) {
                onProgress(i + 1, frameCount);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        await encoder.flush();
        if (encodingError) throw encodingError;
        if (samples.length !== frameCount || !description || !samples[0].key) {
            throw new Error('The encoder did not produce a complete video.');
        }
        return makeBlurMP4(samples, description, capture.width, capture.height, fps, isAV1);
    } finally {
        if (encoder.state !== 'closed') encoder.close();
    }
}

// Minimal ISO BMFF writer for one constant-frame-rate AV1 or AVC video track,
// no audio, no reordered samples, and files below 4 GiB.
function makeBlurMP4(samples, codecConfig, width, height, fps, isAV1 = false) {
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
    const ftyp = box('ftyp', text('isom'), u32(512), text(isAV1 ? 'isomiso2av01mp41' : 'isomiso2avc1mp41'));
    if (dataSize + ftyp.length + 8 >= 0x100000000) throw new Error('Video is too large for this MP4 exporter.');

    // All frames occupy one contiguous chunk, immediately after ftyp + mdat header.
    const stbl = box('stbl',
        fullBox('stsd', 0, u32(1), box(isAV1 ? 'av01' : 'avc1',
            zeros(6), u16(1), zeros(16), u16(width), u16(height),
            u32(0x480000, 0x480000, 0), u16(1), zeros(32), u16(24), u16(0xffff),
            box(isAV1 ? 'av1C' : 'avcC', codecConfig))),
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

// WebCodecs AV1 has no decoderConfig.description. Build av1C from the actual
// sequence header (the encoder may choose a different level than requested).
// Keep the header OBU in av1C as well as in the first sync sample.
// Layout: https://aomediacodec.github.io/av1-isobmff/#av1codecconfigurationbox
function makeBlurAV1Config(data) {
    let start = 0;
    while (start < data.length) {
        const header = data[start];
        let offset = start + 1 + ((header & 4) ? 1 : 0);
        if (!(header & 2)) throw new Error('AV1 OBU is missing its size field.');
        let size = 0;
        let shift = 0;
        let byte;
        do {
            if (offset >= data.length || shift > 49) throw new Error('Invalid AV1 OBU size.');
            byte = data[offset++];
            size += (byte & 127) * 2 ** shift;
            shift += 7;
        } while (byte & 128);
        const end = offset + size;
        if (end > data.length) throw new Error('Truncated AV1 OBU.');
        if (((header >> 3) & 15) !== 1) { start = end; continue; }

        let bit = offset * 8;
        const read = count => {
            if (bit + count > end * 8) throw new Error('Truncated AV1 sequence header.');
            let value = 0;
            for (let i = 0; i < count; i++, bit++) value = value * 2 + ((data[bit >> 3] >> (7 - (bit & 7))) & 1);
            return value;
        };
        const profile = read(3);
        read(1); // still_picture
        const reduced = read(1);
        let level, tier = 0;
        if (reduced) {
            level = read(5);
        } else {
            let decoderModel = 0, delayBits = 0;
            if (read(1)) { // timing_info_present_flag
                read(32); read(32);
                if (read(1)) { // equal_picture_interval: unsigned Exp-Golomb
                    let leading = 0;
                    while (!read(1)) {
                        if (++leading > 31) throw new Error('Invalid AV1 timing information.');
                    }
                    read(leading);
                }
                decoderModel = read(1);
                if (decoderModel) { delayBits = read(5) + 1; read(32); read(5); read(5); }
            }
            const initialDelay = read(1);
            const operatingPoints = read(5) + 1;
            for (let i = 0; i < operatingPoints; i++) {
                read(12);
                const pointLevel = read(5);
                const pointTier = pointLevel > 7 ? read(1) : 0;
                if (i === 0) { level = pointLevel; tier = pointTier; }
                if (decoderModel && read(1)) { read(delayBits); read(delayBits); read(1); }
                if (initialDelay && read(1)) read(4);
            }
        }
        const widthBits = read(4) + 1, heightBits = read(4) + 1;
        read(widthBits); read(heightBits);
        if (!reduced && read(1)) { read(4); read(3); } // frame IDs
        read(1); read(1); read(1); // superblock size, filter intra, intra edge filter
        if (!reduced) {
            read(1); read(1); read(1); read(1); // inter-intra, masked compound, warped motion, dual filter
            const orderHint = read(1);
            if (orderHint) { read(1); read(1); }
            const screenContent = read(1) ? 2 : read(1);
            if (screenContent > 0 && !read(1)) read(1);
            if (orderHint) read(3);
        }
        read(1); read(1); read(1); // super-resolution, CDEF, restoration
        const highBitdepth = read(1);
        const twelveBit = profile === 2 && highBitdepth ? read(1) : 0;
        const monochrome = profile === 1 ? 0 : read(1);
        let primaries = 2, transfer = 2, matrix = 2;
        if (read(1)) { primaries = read(8); transfer = read(8); matrix = read(8); }
        let subX = 0, subY = 0, position = 0;
        if (monochrome) {
            read(1);
            subX = subY = 1;
        } else if (!(primaries === 1 && transfer === 13 && matrix === 0)) {
            read(1); // color_range
            if (profile === 0) { subX = subY = 1; }
            else if (profile === 2) {
                subX = twelveBit ? read(1) : 1;
                subY = twelveBit && subX ? read(1) : 0;
            }
            if (subX && subY) position = read(2);
        }
        const config = new Uint8Array(4 + end - start);
        config.set([0x81, (profile << 5) | level,
            (tier << 7) | (highBitdepth << 6) | (twelveBit << 5) |
            (monochrome << 4) | (subX << 3) | (subY << 2) | position, 0]);
        config.set(data.subarray(start, end), 4);
        return config;
    }
    throw new Error('AV1 key frame is missing its sequence header.');
}

// Queue-drained wall-clock measurements include CPU submission and GPU idle time.
// Revisit every radius on each sweep to spread clock/thermal drift across points.
async function benchmarkBlur({ device, methods, setCase, renderFrame, onProgress, isCancelled }) {
    const radii = [];
    for (let r = 5; r < 1000 / 1.1; r *= 1.2) radii.push(Math.round(r));
    // Merge the last near-endpoint sample into 1000 to keep labels readable.
    radii.push(1000);
    const series = methods.map(method => ({ ...method, points: radii.map(radius => ({ radius, samples: [], iterations: 1 })) }));
    const sweeps = 4;
    let lastProgressUpdate = -Infinity;
    const check = () => { if (isCancelled()) throw new Error('Benchmark cancelled.'); };
    async function batch(iterations) {
        await device.queue.onSubmittedWorkDone();
        check();
        const start = performance.now();
        for (let i = 0; i < iterations; ++i) await renderFrame();
        await device.queue.onSubmittedWorkDone();
        return (performance.now() - start) / iterations;
    }
    for (let sweep = 0; sweep <= sweeps; ++sweep) {
        for (let i = 0; i < radii.length; ++i) {
            // Rotate method order so one method is not always first.
            for (let m = 0; m < series.length; ++m) {
                check();
                const result = series[(m + sweep) % series.length];
                const point = result.points[i];
                if (performance.now() - lastProgressUpdate >= 200) {
                    onProgress(`${sweep === 0 ? 'Warm-up' : `Sweep ${sweep}/${sweeps}`} · ${result.name} · radius ${point.radius}`);
                    lastProgressUpdate = performance.now();
                    // Yield for progress painting/cancellation outside the timed batch.
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
                setCase(result.mode, point.radius);
                await renderFrame();
                await device.queue.onSubmittedWorkDone();
                const ms = await batch(sweep === 0 ? 2 : point.iterations);
                if (sweep === 0) {
                    // Aim for 30 ms batches; bound queued work on fast devices.
                    point.iterations = Math.max(1, Math.min(128, Math.ceil(30 / Math.max(ms, 0.01))));
                } else {
                    point.samples.push(ms);
                }
            }
        }
    }
    check();
    for (const result of series) for (const point of result.points) point.ms = Math.min(...point.samples);
    return { series, sweeps };
}

function makeBenchmarkSVG({ series, sweeps }, width, height) {
    const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
    const values = series.flatMap(s => s.points.map(p => p.ms));
    if (values.some(v => !Number.isFinite(v) || v <= 0)) throw new Error('Invalid benchmark timing.');
    const minimum = Math.min(...values);
    const low = minimum >= 0.05 ? 0.05 : 10 ** Math.floor(Math.log10(minimum / 1.2));
    const high = Math.max(1, Math.max(...values) * 1.6);
    const left = 95, right = 1040, top = 70, bottom = 480;
    const x = radius => left + Math.log(radius / 5) / Math.log(1000 / 5) * (right - left);
    const y = ms => bottom - Math.log(ms / low) / Math.log(high / low) * (bottom - top);
    const parts = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 ${series.length > 5 ? 610 : 580}" role="img" aria-labelledby="title desc">`,
        `<title id="title">Blur timings (${width}x${height})</title>`,
        `<desc id="desc">Minimum of ${sweeps} batch averages. CPU and GPU wall-clock milliseconds per frame; logarithmic axes. No display or video encoding.</desc>`,
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#333">',
        `<text x="95" y="30" font-size="20">Blur timings (${width}x${height})</text>`];
    // Embed the actual samples and batch sizes for reproducibility.
    parts.push(`<metadata>${escape(JSON.stringify({ width, height, sweeps, series }))}</metadata>`);
    for (let e = Math.floor(Math.log10(low)); e <= Math.ceil(Math.log10(high)); ++e) {
        for (const multiplier of [1, 2, 5]) {
            const tick = multiplier * 10 ** e;
            if (tick < low || tick > high) continue;
            parts.push(`<path d="M${left} ${y(tick)}H${right}" stroke="#e5e5e5"/>`,
                `<text x="85" y="${y(tick) + 4}" text-anchor="end" font-size="12">${Number(tick.toPrecision(4))}</text>`);
        }
    }
    for (const { radius } of series[0].points) {
        parts.push(`<path d="M${x(radius)} ${top}V${bottom}" stroke="#eee"/>`,
            `<text x="${x(radius)}" y="501" text-anchor="middle" font-size="12">${radius}</text>`);
    }
    parts.push('<text x="567.5" y="528" text-anchor="middle" font-size="15">Blur radius (pixels)</text>',
        '<text transform="translate(25 275) rotate(-90)" text-anchor="middle" font-size="15">Time, ms</text>');
    const labels = [];
    for (const result of series) {
        parts.push(`<polyline points="${result.points.map(p => `${x(p.radius)},${y(p.ms)}`).join(' ')}" fill="none" stroke="${result.color}" stroke-width="${result.emphasized ? 4 : 2.5}"/>`);
    }
    // Place labels together by radius, separating close methods vertically.
    for (let i = 0; i < series[0].points.length; ++i) {
        const column = series.map(s => ({ s, p: s.points[i], labelY: y(s.points[i].ms) - 10 })).sort((a, b) => a.labelY - b.labelY);
        for (let j = 1; j < column.length; ++j) column[j].labelY = Math.max(column[j].labelY, column[j - 1].labelY + 16);
        const overflow = Math.max(0, column.at(-1).labelY - (bottom - 5));
        for (const { s, p, labelY } of column) {
            const px = x(p.radius), py = y(p.ms), ly = labelY - overflow;
            parts.push(`<circle cx="${px}" cy="${py}" r="4" fill="${s.color}"><title>${escape(s.name)}: radius ${p.radius}, ${p.ms.toFixed(2)} ms</title></circle>`);
            // Keep the first column clear of the Y-axis tick labels.
            const anchor = i === 0 ? 'start' : 'middle';
            labels.push(`<text x="${px}" y="${ly}" text-anchor="${anchor}" fill="${s.color}" stroke="white" stroke-width="3" paint-order="stroke" font-size="11" font-weight="${s.emphasized ? 700 : 400}">${p.ms.toFixed(2)}</text>`);
        }
    }
    parts.push(...labels);
    series.forEach((s, i) => {
        const columns = series.length > 5 ? 3 : series.length;
        const lx = 95 + (i % columns) * (1000 / columns);
        const ly = 557 + Math.floor(i / columns) * 28;
        parts.push(`<path d="M${lx} ${ly}h28" stroke="${s.color}" stroke-width="${s.emphasized ? 4 : 2.5}"/><text x="${lx + 38}" y="${ly + 5}" font-size="15" font-weight="${s.emphasized ? 700 : 400}">${escape(s.name)}</text>`);
    });
    parts.push('</g></svg>');
    return parts.join('\n');
}

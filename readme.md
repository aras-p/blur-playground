# Blur Playground

JavaScript/WebGPU implementation of several image blurring algorithms. Runs in the browser
-- open `index.html` (requires WebGPU capable browser with `float32-filterable` and
`float32-blendable` features).

In order to load provided sample image files, just opening the HTML page
in a browser won't work. Easiest is then to run `python3 -m http.server 8000`
and go to `http://localhost:8000/index.html`.

### Code layout

`index.html` contains the UI, image loading, and rendering orchestration.
Blur shaders, pipelines, and algorithm helpers live in `js/`:

- `separable-blur.js`: Box, Tent, and Gaussian; Fast Gaussian also uses this for small radii.
- `smol-gaussian.js`: Smol Gaussian.
- `ryg-blur.js`: repeated fractional box filters evaluated with moving sums.
- `dual-kawase.js`: Dual Kawase.
- `fast-gaussian.js` and `skia-gaussian.js`: Fast and Skia Gaussian.
- `gpu-helpers.js`: shared fullscreen vertex shader, GPU resource helpers, and texture cache.
- `video-export.js`: WebCodecs H.264 encoding and a minimal MP4 writer.
- `exrloader.js` and `fflate.js`: EXR decoding and decompression.

The files use classic scripts with shared globals; GPU helpers load before the
blur implementations. Most pipelines are initialized after device creation;
Fast Gaussian compute pipelines are created on first use.

### Blur Algorithms

- **Box**, **Tent**, **Gaussian** - separable blurs
  implemented with the same shader, just different convolution
  kernel shapes. Box and Tent are not great blurs; just here
  because they were easy to do.
- **Fast Gaussian** - Blender compositor's recursive Gaussian blur: direct convolution
  for small radii, Deriche for medium radii, and parallel second-order Van Vliet
  sections for large radii. See details below.
- **Ryg Blur** - repeated fractional box filters using moving sums, based on
  Fabian Giesen’s “Fast blurs” posts. Adjustable 1–5 boxes per axis (default 3);
  see the GPU implementation and precision tradeoffs below.
- **Smol Gaussian** - a Gaussian approximation using downsampling, small separable
  filters, and reconstruction, with smooth transitions between working resolutions.
- **Skia Gaussian** - Skia’s GPU image-filter approach: progressive bilinear
  downsampling, a small Gaussian, and bilinear reconstruction. Supports independent
  X/Y radii; see implementation details below.
- **Dual Kawase** - multi-pass downsample/upsample pyramid blur, from
  Marius Bjørge, [Bandwidth-Efficient Rendering](https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf) (SIGGRAPH 2015),
  see also explanation in [this blog post](https://blog.frost.kiwi/dual-kawase/#dual-kawase-blur).
  This is very fast for large blurs. The original algorithm blurs the same amount
  horizontally & vertically and only allows very discrete
  amounts of blur (more or less "powers of two" radii), so to achieve arbitrary radius there's
  an extra step that blends between two discrete blur amounts, similar to how [obs-composite-blur](https://github.com/FiniteSingularity/obs-composite-blur) does it.

### Fast Gaussian

Ported from the local Blender checkout at commit
`02c927c0ffd87e08aca64ceed572b5ff20556b1e`, specifically the compositor's
`recursive_gaussian_blur.cc`, Deriche/Van Vliet coefficient helpers, and GPU shaders.
The larger X/Y radius selects the method for **both** axes: direct convolution
below 9, fourth-order Deriche from 9 to below 96, and Van Vliet from 96 onward.
Van Vliet uses Blender's pole scaling and partial-fraction decomposition into
four parallel causal/non-causal second-order filters, not a cascade from the paper.

Recursive passes use `sigma = max(radius, 1) / 3` per axis, including an axis
set to zero, matching Blender. Both radii zero bypass blur. Boundary pixels
extend indefinitely (Blender's non-extended-bounds mode); the initial recursive
state uses Blender's boundary coefficients. Independent row scans are followed
by a sum-and-transpose pass for each axis.

Textures retain the playground's RGBA32F format rather than Blender's default
RGBA16F intermediates, so this is an algorithm match, not a bit-for-bit match.
Single-precision recursive arithmetic can drift, especially just below the
Deriche/Van Vliet switch; browser/GPU compiler choices also affect rounding.
The filter retains Blender's approximations, including possible small negative
lobes and changes at algorithm thresholds.

### Ryg Blur

Based on Fabian Giesen’s [Fast blurs 1](https://fgiesen.wordpress.com/2012/07/30/fast-blurs-1/)
and [Fast blurs 2](https://fgiesen.wordpress.com/2012/08/01/fast-blurs-2/).
`js/ryg-blur.js` uses a compute invocation per scanline with a moving sum,
using two hardware-filtered samples per update for fractional box endpoints.
This trades interpolation precision for fewer sampling instructions: sampler
subtexel weight precision can cause recurrence errors, especially at small radii.
Initialization uses exact texture loads. Boundaries
clamp to the edge; sums and intermediate textures use 32-bit floats.

**Iteratons (ryg)** under **Blur Options** selects 1–5 boxes per axis (default 3), enabled only
for this method. Each box’s exact discrete variance is matched to
`(radius / 3)² / count`, so increasing count changes the approximation’s shape
while preserving its variance. Zero-radius axes are skipped. The benchmark
uses the selected count, records it in SVG metadata, and plots Ryg in teal
before Smol Gaussian. Its scanline dependencies limit GPU parallelism, so
performance should be judged using the benchmark rather than assumed from
its low sample count. Initialization still costs O(box radius) per scanline.

### Smol Gaussian

Radius maps approximately to three Gaussian standard deviations, like the existing
Gaussian mode. Each axis reduces independently to keep the working sigma small;
a zero-radius axis keeps its original resolution and is not filtered. Exact 2×
downsampling uses one bilinear sample, including when only one axis is reduced;
odd-sized reductions integrate pixel areas to preserve bright points. The residual
Gaussian compensates approximately for reduction and reconstruction variance.
For an axis reduced by a factor `s`, its working sigma is
`sqrt(max(0, targetSigma² - reductionVariance - reconstructionVariance)) / s`.
The variance terms are measured in original-image pixels squared. This avoids
adding a full-width Gaussian on top of blur already introduced by resizing;
resampling phase and boundary effects make the compensation approximate.

Each reduced axis has a one-texel border on both sides. Reduction preserves the
source edge/corner values there instead of extending averaged interior pixels;
this avoids bright interiors contaminating the clamped boundary at large radii.
The other axis still integrates its footprint along each edge. Padding is carried
through every reduction and filtered with the image, then excluded from the
logical dimensions used for sigma, level selection, and reconstruction coordinates.
Unreduced axes need no padding. This follows the same boundary-preservation idea
as Skia, while retaining the area reductions and reconstruction filters below.
For example, an 8×5 logical level occupies a 10×7 texture when both axes reduce.
This adds border pixels and coordinate calculations, but no additional render
passes; exact 2× reductions still use one bilinear sample per output pixel.

Eligible pairs of exact 2× reductions are fused into one 4× reduction per
reduced axis. Four bilinear reads cover a 4×4 footprint, or two cover a 4×1
footprint for a single-axis reduction. The planner works from the source forward
to eliminate the largest intermediates first. It keeps levels needed by active
blur endpoints or shared branches, and requires each reduced source dimension
to be exactly divisible by four; odd-size steps keep the original area filter.
The preserved borders, working-resolution choices, and variance calculation stay
the same. The fused filter matches two exact halves apart from filtering and
intermediate rounding.

For example, at radius X=Y=1000, 1920×1080 starts with
`1920×1080 → 480×270 → 240×135`, skipping the 960×540 intermediate.
At radius X=Y=500, 3840×2160 starts with two fused steps:
`3840×2160 → 960×540 → 240×135`. These are logical sizes, excluding padding.
Each fusion removes one render pass and its intermediate write/read; actual
speedup depends on the GPU and the rest of the blur. **No 4x reduction (smol)**
is unchecked by default; checking it bypasses fusion planning and restores only
2× reduction steps. The benchmark respects this option, while enabling level
blending, so it can be used to compare the two reduction paths.

Reconstruction chooses per axis: bilinear for enlargement up to 2× (including
odd-sized half-resolution images), positive cubic B-spline for coarser levels.
This needs one bilinear sample when both axes use bilinear, two when only one
needs cubic, and four when both do. Cubic removes coarse-grid slope discontinuities
without ringing around HDR highlights. Variance compensation follows the chosen
filter; unreduced axes preserve sharp detail. Filter choice depends on endpoint
dimensions, so the existing crossfade also blends changes in reconstruction.
An `ALL_BILINEAR` WebGPU pipeline constant specializes reconstruction when all
contributing endpoints use bilinear on both axes; other draws use the general
hybrid shader. Both variants are created during initialization.

Gaussian samples are paired using bilinear filtering, with at most 25 texture
reads per pixel per axis. Textures and uniform buffers are reused. Reduction
prefixes are shared between transition paths, and reconstruction writes the
full-resolution output once. When no resizing or blending is needed, the final
Gaussian pass writes directly to the output and reconstruction is skipped.

Before an axis changes resolution, a smoothstep crossfade blends two approximations
of the **same target blur**. The blending range is **sigma 5–6 in the current
level's texels**, before variance compensation: `sigma = radius / 3`, divided by
the actual reduction scale for that axis. Blending starts at 5 and the axis
switches to the next level at 6; below 5, neighboring endpoints are skipped.
The **No level blend (smol)** checkbox skips the crossfade entirely.
Transitions use triangular interpolation: at most three small Gaussian results,
and only two when the X/Y transition fractions match. A canonical reduction tree
reduces both axes together first, then any remaining axis, so neighboring endpoints
share the expensive large-image downsample prefix. The fixed route also keeps
odd-sized resampling consistent across transition boundaries. Each is reconstructed directly onto the output grid to avoid an extra
resize appearing or disappearing at a boundary. Kernel tails taper smoothly as
the tap count changes. This prioritizes smooth radius changes and a rounded blur
shape over exact Gaussian matching; resampling still introduces some phase-dependent
shape variation. Radius sliders use steps of 1.

Area reduction, separable Gaussian filtering, and reconstruction are the building
blocks. The variance estimates, transition thresholds, tapered tails, and choice
of three endpoints define this particular approximation. Unlike Dual Kawase's
blend between fixed blur widths, Smol Gaussian's endpoints all target the
same requested width; their blend hides a change of working resolution.

Related work for the building blocks:

- Fabian Giesen, [Gaussian blur kernels (2009)](https://sourceforge.net/p/gdalgorithms/mailman/message/23077758/):
  low-pass filtering before downsampling, a main blur at reduced resolution,
  and reconstruction afterward. His warning about thin objects flickering when
  downsampling skips samples motivates careful reduction, though his suggested
  prefilters differ from our area integration.
- Cornell CS5625, [Apply blur to mipmap levels (2022)](https://www.cs.cornell.edu/courses/cs5625/2022sp/assignments/pipeline.html#323-apply-blur-to-mipmap-levels):
  an explicit recipe to downsample by `2^k`, blur with sigma `sigma / 2^k`, and
  upsample. This describes the basic structure of one endpoint before our
  variance compensation. Its bloom merge combines different blur widths,
  rather than alternative resolutions of the same target width.
- Intel, [An Investigation of Fast Real-Time GPU-Based Image Blur Algorithms](https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html),
  **Working in Lower Resolution**: downsample, apply a smaller Gaussian, then upscale.
- Sigg and Hadwiger, [GPU Gems 2, Chapter 20: Fast Third-Order Texture Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering):
  cubic B-spline reconstruction using paired hardware-linear samples.
- Bjørge, [Bandwidth-Efficient Rendering, SIGGRAPH 2015](https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf):
  filtering across multiple resolutions. Its mixed-resolution pipeline is distinct
  from crossfading alternative grids for the same target blur here.

These sources explain individual techniques, not the exact Smol Gaussian
combination implemented in this playground.

### Skia Gaussian

Adapted from Google Skia revision `da51f0d60e` (this is a WebGPU re-implementation
of the algorithm, not a Skia wrapper).
The relevant source is [SkImageFilterTypes.cpp](https://github.com/google/skia/blob/da51f0d60e/src/core/SkImageFilterTypes.cpp)
(`FilterResult::rescale` / `Builder::blur`) and
[SkBlurEngine.cpp](https://github.com/google/skia/blob/da51f0d60e/src/core/SkBlurEngine.cpp)
(`SkShaderBlurAlgorithm`).

- Radius maps to sigma = radius / 3; sigma at or below 0.03 bypasses that axis.
- Each axis reduces independently to a working sigma of at most 4. Intermediate
  steps halve the scale; the last step uses the remaining fractional scale.
  Skia's near-identity final-step collapse is preserved.
- Logical bounds remain fractional and scaling is centered. A one-pixel border
  preserves clamped edge/corner values through downsampling, including odd sizes.
- The normalized Gaussian has radius ceil(3 × sigma). Small 2D kernels (up to
  28 samples) use one pass; others use separable, bilinear-paired 1D taps.
- A single bilinear upscale reconstructs the result. There is no variance
  compensation, cubic reconstruction, tap taper, or transition crossfade.
  This deliberately retains Skia's radius-dependent approximation changes.

Only whole-image blur with clamped edges is implemented. The existing HDR
RGBA32F texture format is retained, whereas the native Metal reference uses
RGBA16F. Half-float rounding and GPU sampler precision mean results are not
bit-identical. Temporary textures and uniforms are reused, with the texture pool
bounded to the current pass chain rather than every size visited during animation.

### Dual Kawase interpolation

Dual Kawase is the main focus of this playground. The basic algorithm works well;
the implementation extends it in two ways:

- **Arbitrary blur sizes:** between discrete blur steps, linearly blend the
  previous and next blur amounts using the fractional position between steps
  remapped with `t * (2 + t) / 3`. This compensates for blending kernel variance
  rather than width, making blur growth more even between doubling steps.
  Radii are not rounded before
  interpolation, and the filters use the normal discrete-step sampling offsets.
- **Independent X/Y blur radii:** interpolate up to three neighboring discrete
  blur results, using triangles in the X/Y radius grid. At discrete radii,
  the corresponding kernel is used directly; neighboring triangles share
  their boundary results.

For equal X/Y radii, only two isotropic endpoints contribute. Their shared
downsampling and upsampling run once, with one extra down/up pair and a blend
at the smallest shared level. Anisotropic interpolation also shares pyramid work and blends
before the common upsampling, but can require additional passes. Interpolation
is continuous; its slope can change at triangle and discrete-step boundaries.
Interpolation uses a constant blend weight for all RGBA channels, preserving alpha.

The remapping can be motivated by assuming neighboring kernel widths are `r`
and `2r`. Variance interpolation gives a weight of
`((r * (1 + t))² - r²) / ((2r)² - r²) = t * (2 + t) / 3`.
Actual Kawase kernels only approximately follow this model, and the first
interval (radius 0 to 6) uses the same remapping as a heuristic. The radius scale
is an approximate visual match to Gaussian blur, not an exact sigma calibration.

For remapped X/Y fractions `tx` and `ty`, the three weights are
`1 - max(tx, ty)`, `abs(tx - ty)`, and `min(tx, ty)`. The middle endpoint advances
the axis with the larger fraction; the other two advance neither or both.
These nonnegative weights sum to one and agree along shared triangle edges.
Blending before the common upsample suffix is valid because that fixed filter
is linear: `U(sum(w_i * image_i)) = sum(w_i * U(image_i))`. This requires the
same grids and sampling path for the shared suffix, including on odd-sized images.

### Performance

Benchmark checkboxes select which methods to measure (all selected by default).
Deselected methods are skipped; remaining methods retain their chart colors and
Smol Gaussian remains last. Select at least one method before starting.

**Benchmark** measures Gaussian, Fast Gaussian, Dual Kawase, Skia Gaussian, Ryg Blur, and Smol Gaussian
on the loaded image with equal X/Y radii, from 5 to 1000 at approximately 1.2×
spacing. After a warm-up sweep, it runs four sweeps and plots the minimum
batch-average milliseconds per frame at each radius. Each batch waits for the
WebGPU queue before and after timing; CPU command preparation, GPU execution,
and GPU idle time are included. Display, UI updates, and video encoding are excluded.
Smol blending is enabled for these comparisons. Batches target 30 ms using the
warm-up estimate, with 1–128 renders per batch.

The SVG is displayed directly below the image, with a Download SVG link. It has
logarithmic axes and two-decimal timing labels. Progress updates at most every
200 ms, outside the timed batches. Its
Y axis starts at 0.05 ms, or lower when needed to show faster results. Raw samples
and batch sizes are embedded in the SVG metadata. Cancel stops after the current
batch; the original blur settings are restored. **Render Video** exports video
without reporting a benchmark time.

The historical measurements below used the previous video-export timer and are
not directly comparable to the dedicated benchmark.

**Render Video** time in seconds, doing an animated radius sweep on a 1920x1080 input image,
all on Chrome browser:

| Scenario | Apple M4 Max | RTX 3080Ti, Windows | Intel Iris Xe, Windows |
|----------|-------------:|------:|-------:|
|No blur          |  1.41 |  0.98 |   2.71 |
|Gaussian         | 19.52 | 15.47 | 151.07 |
|Fast Gaussian    |  2.25 |  1.86 |  10.09 |
|**Smol Gaussian**|  1.55 |  1.11 |   3.63 |
|Skia             |  1.47 |  1.15 |   3.58 |
|Dual Kawase      |  1.72 |  1.23 |   4.89 |


### Features

- Drag & drop or browse for images (PNG/JPG/EXR).
- Adjustable blur radius (X/Y can be locked or independent).
- Inspect the source, intermediate textures, or final output with **Display Texture**.
- **Animate** loops an eight-second radius sweep, respecting the X/Y lock.
- **Render Video** exports an eight-second, 60 fps H.264 MP4 using WebCodecs,
  reduced to 960px on the longer axis. Blur itself stays at the image resolution.

### External code

- `js/skia-gaussian.js` adapts Skia algorithms (Google LLC, BSD-3-Clause);
  see `js/skia-LICENSE.txt`.

- `js/fast-gaussian.js` is adapted from Blender compositor code
  (Blender Authors, GPL-2.0-or-later); see its source header.

- `js/exrloader.js` is EXR file loader, adapted from three.js
  ([file link](https://github.com/mrdoob/three.js/blob/65bfbd8e51db/examples/jsm/loaders/EXRLoader.js))
  which itself is adapted from [tinyexr](https://github.com/syoyo/tinyexr).
- `js/fflate.js` is gzip/deflate decoder needed for EXR loading,
  from [fflate](https://101arrowz.github.io/fflate/).

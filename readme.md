# Blur Playground and the "Smol Gaussian" blur

This is a JavaScript/WebGPU toy for comparing image blurring algorithms.
Primary interest is perhaps the **Smol Gaussian** algorithm: a Gaussian
approximation that is efficient at large blurs, supports smoothly varying
blur radius and different X/Y blur amounts. It is somewhat similar to
GPU blur used in Skia, with several quality improvements.

A handful of other blurs are also implemented here: regular separable Gaussian,
"recursive" Deriche / Van Vliet, blur from Skia, Dual Kawase with extensions
to support arbitrary blur sizes and different X/Y radii, and repeated box blurs
approach from Fabian Giesen's blog posts.

Requires WebGPU support (including `float32-filterable` and
`float32-blendable`), you can just open the `index.html`, or serve it locally
like `python3 -m http.server 8000` and then go to `http://localhost:8000`. The latter
approach is needed if you want to be able to load `test_a_small` and `test_b_1080p`
sample files with a single click in the page.

- Pick any of the six blur modes! *For free!*
- Drag & drop or browse for images (PNG, JPG and a subset of EXR).
- Adjustable blur radius, with ability to lock X/Y.
- Can inspect various intermediate textures produced by a blurring algorithm.
- `Animate` checkbox animates the radius and the source image rotation.
- `Render Video` button exports animated blur result into a video file, using the current blur mode.
  The video is encoded using WebCodecs browser functionality, and resized to max 960px. Blur itself
  is performed at full image resolution.
- `Benchmark` button tests increasing blur radii with all the selected blur algorithms,
  and produces a SVG file with the graphs. The result is displayed at the bottom of the page,
  and can be downloaded too.

## Blur Modes

In the videos below, blur radius is animated (increase X&Y together from 5 to 1000, decrease X&Y separately),
and the input image is rotated during the animation loop. The input image has very bright
objects (500 and 50 intensity in linear), here's the input with additional glare to show "hey I'm bright"
(actual image used for tests has no glare).

<img src="videos/input_with_glare.png" width="400">

### Smol Gaussian

"Downsampled" Gaussian blur: downsample to a working resolution (which could be different
per axis), apply a small separable Gaussian there, reconstruct to full resolution image.

<img src="videos/blur_b_smol_gaussian.avif" width="400">

The algorithm is similar to Skia's GPU Gaussian blur as of 2026 Sep (`FilterResult::Builder::blur`
and `FilterResult::rescale` in [SkImageFilterTypes.cpp](https://skia.googlesource.com/skia/+/15a9437eec87/src/core/SkImageFilterTypes.cpp)) - independent X/Y scaling, texture samples placed to use
bilinear filtering, one pixel border on downsampled images that preserve the original
image edges (this way bright interiors do not overbright the result at large radius).

Additions compared to Skia blur are:

- When downsampling odd-sized image, we do more correct pixel area integration,
  so that isolated bright pixels do not flicker with sampling phase shift.
- Downsampled levels are ceil-halved sizes, and downsampling kicks in at sigma=6
  (so in practice, blur sizes 18, 36, 72, 144, ... switch to new level). Skia instead
  scales continuously, to keep working sigma under 4.
- When doing the final Gaussian blur, we take into account the blur introduced by downsampling
  and later reconstruction, i.e. subtract their variance from the blur kernel.
- The final Gaussian kernel extends to sigma=4 (i.e. not truncated at sigma=3), and
  weights in sigma 3..4 region are tapered to reach zero. This helps to reduce the
  "blur is cut off" with very bright highlights, and looks better when radius is animated
  and number of taps changes.
- Final reconstruction to full image size uses a cubic B-spline (fully positive kernel, so no
  ringing) instead of bilinear, beyond 2x enlargement. This helps to avoid slope discontinuities
  of bilinear.
- Pairs of exact 2x reductions are done as single 4x reduction as an optimization.

Related work for all of this:
- Skia [SkImageFilterTypes.cpp](https://skia.googlesource.com/skia/+/15a9437eec87/src/core/SkImageFilterTypes.cpp), as mentioned above.
- Fabian Giesen, ["Gaussian blur kernels" on gdalgorithms](https://sourceforge.net/p/gdalgorithms/mailman/message/23077758/) list (2009): low-pass filter,
  blur at lower resolution, then upsample. He suggests bilinear upsampling is good enough, however
  in my tests very bright (HDR) blurred objects show bilinear "slope steps",
  so I used cubic.
- Intel, "[An Investigation of Fast Real-Time GPU-Based Image Blur Algorithms](https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html)"
  (2014) has "Working in Lower Resolution" section, but with not much details at when you would switch to it,
  or how to handle animated blur radius.
- GPU Gems 2, Chapter 20, "[Fast Third-Order Texture Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering)"
  has a trick for cubic B-spline filtering using bilinear samples, which is what I used
  here in reconstruction part.

<details>
<summary>Discarded idea - crossfading working resolutions:</summary>

I tried blending neighboring resolutions to hide level switches. Idea was this:
when the resolution that does the final blur changes, in theory you could have
a visible "jump" if blur radius is animated.

So if resolution switch happens at sigma=6, then starting at sigma=5 already, do
both the current resolution and the next resolution, and blend between them using
a smoothstep curve. Both evaluations target the same final blur amount, just
at different grid, and so each of them uses different amount of gaussian taps.

This however cost in performance, since now during transition regions (blur sizes
15..18, 30..36, 70..72 etc.) there are two Gaussian blurs performed and their
result is blended. If the X/Y radii are different, we might need to evaluate
three Gaussian blurs in fact, and blend between them.

In my testing, this brought pretty much no visual difference, but cost in
performance and made the implementation more complex. So eventually this was discarded.
</details>


### Dual Kawase

This project also implements an extended version of Dual Kawase. See Marius Bjørge,
[Bandwidth-Efficient Rendering](https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf) (SIGGRAPH 2015),
also explanation in [this blog post](https://blog.frost.kiwi/dual-kawase/#dual-kawase-blur).
The original uses equal horizontal/vertical blur amounts, and only supports a discrete "number of blur pyramid levels"
control, not a continuous "blur radius" setting. 

For arbitrary blur sizes: blend between neighboring discrete blur levels, similar to
[obs-composite-blur](https://github.com/FiniteSingularity/obs-composite-blur).
Blend using the fractional position between steps
remapped with `t * (2 + t) / 3`, this makes it feel a bit nicer than just a linear blend.

For independent X/Y blur radii: when the smaller blur radius is reached, we stop
further reductions along that axis. For the between-levels blend above, we might need
to blend between three different blurred results.

<img src="videos/blur_b_dual_kawase.avif" width="400">

The result kinda works, but does not "feel great" to me. Smoothly animated blur radius does not "feel" smooth on HDR
highlights, due to how blending happens between discrete blur levels.

### Skia Gaussian

This is what conceptually is closest to "Smol Gaussian". Implementation here
is just a WebGPU re-implementation of relevant parts of Skia `SkImageFilterTypes.cpp`, as it
was in revision `15a9437eec87` (2026 Sep). Basic algorithm is:

- Each axis independently downscales to a working sigma `<= 4`. Intermediate steps
  halve the scale, and the last step uses the remaining fractional scale.  
- A one pixel border around intermediate steps preserves clamped edge colors.
- Final Gaussian pass has radius of `ceil(sigma * 3)`. A single direct convolution,
  or separable bilinear-paired filter samples is used depending on size.
- Final result is bilinearly upscaled to original resolution.

<img src="videos/blur_b_skia_gaussian.avif" width="400">

Skia blur feels just fine on regular LDR content, but on very bright HDR highlights, the aliasing and wobbling
are quite apparent.

### "Fast Gaussian" (from Blender 5.2)

Blender's compositor blur node got a "Fast Gaussian" mode back in 2008 (v2.46)
in [2a2453d3](https://projects.blender.org/blender/blender/commit/2a2453d3),
which however built upon earlier implemented `IIR_Gauss` functionality for defocus
blur node (2006, v2.43, commit [e61dec07](https://projects.blender.org/blender/blender/commit/e61dec07)).

This builds upon "Recursive Gaussian Filtering", which if you're just a programmer
but not familiar with signal processing terminology, is *quite a confusing* name.
You'd think a recursive gaussian would be something about building like several smaller versions
and somehow combining them, right? Haha nope, not at all, "recursive filter" in signal processing
just means that the filter uses some of the previous outputs.

Anyway, the "fast" part is due to filter construction that is basically the same cost,
no matter the blur radius. Original code in Blender seemingly was built
on one such algorithm, from Young, van Vliet & van Ginkel, “Recursive Gabor Filtering” (2000)
paper. Many years later, in 2024 (blender 4.2.0, commit [382131fe](https://projects.blender.org/blender/blender/commit/382131fe))
Omar remade this algorithm to have both CPU and GPU
code paths, and to avoid double precision. And it was built on a handful of papers, curiously
enough an *earlier* paper by Young, van Vliet et al. "Recursive Gaussian derivative filters" (1998),
another paper Deriche "Recursively implementing the Gaussian and its derivatives" (1993), and some more.

Anyhoo, in this project there's a WebGPU re-implementation of Blender 5.2 "Fast Gaussian" state (mostly `recursive_gaussian_blur.cc`),
which is fourth order Deriche formulation for radius under 96, and Van Vliet formulation for larger radius.

The algorithms are more or less constant work independent of the blur radius, which is very nice. However, they are also from
25+ years ago, and are not "embarrassingly parallel" that would fit a GPU (or even a many-core CPU) very well. So despite the
name, they may or might not be very _fast_ :)

They also have some ringing artifacts, which are not that much noticeable in regular colors, but with very bright HDR highlights,
blurred result can have halos or negative colors, which is not great. See:

<img src="videos/blur_b_fast_gaussian.avif" width="400">


### Ryg Blur

I'm calling this "Ryg Blur" due to Fabian Giesen's [Fast blurs 1](https://fgiesen.wordpress.com/2012/07/30/fast-blurs-1/)
and [Fast blurs 2](https://fgiesen.wordpress.com/2012/08/01/fast-blurs-2/) blog posts, which nicely
described the whole idea, including how exactly to handle fractional samples at the ends,
and how trivially that extends to a compute shader implementation.

However the idea itself is "repeated box convolution", and has been around for ages, e.g.
Heckbert "[Fun With Gaussians](https://www.researchgate.net/publication/2313072_Fun_With_Gaussians)" (1985)
talk about it in pages 11-12.

> "convolution of a 500x500 image with a 35x35 kernel would take over an hour
> with 2-D convolution, but only 4 minutes with two 1-D convolutions" where he's
> talking about a separable Gaussian filter. Only four minutes to blur a 500x500
> image, imagine that! Computers have gotten quite a bit faster, eh.

Anyway, this implementation is basically fixed cost independent of blur size,
and uses two bilinear samples for fractional box endpoint updates. Number of iterations
control is for how many times this box convolution should be done (1: box filter, 2: tent filter,
3 and up: approaching Gaussian). It is very simple to implement, however not the fastest.
Amount of parallelism (parallel over rows or columns) is nowhere near enough to feed modern
GPUs, and multiple passes over the full size image incur a lot of memory traffic.

<img src="videos/blur_b_ryg.avif" width="400">

### Regular Gaussian blur

A simple separable Gaussian blur, mostly included as a reference. This is one algorithm that
becomes impractical at very large blur radii, since the cost scales linearly with radius.
This particular implementation cuts off the kernel at sigma=3 (matches behavior of Blender),
which is fine for regular image content, but for very bright HDR highlights it makes
the blur feel like it "stops" abruptly. Smol Gaussian feels better in this regard!

<img src="videos/blur_b_gaussian.avif" width="400">


## Performance

All of the below are in Chrome browser. The timings finish any GPU work, start measuring,
do the blur, finish GPU work again, end measuring. So this kinda measures "latency" of doing
both CPU & GPU work parts.

Apple M4 Max, macOS:
![mac m4 chart](/charts/mac-benchmark-3840x2160.svg?raw=true "mac m4 chart")

Ryzen 5950X, RTX 3080Ti, Windows:
![rtx3080 chart](/charts/rtx-benchmark-3840x2160.svg?raw=true "rtx3080 chart")

Core i7-1185G7, Intel Iris Xe, Windows:
![intel xe chart](/charts/xe-benchmark-3840x2160.svg?raw=true "intel xe chart")


## Code layout

`index.html` is the main file, with UI, image loading, WebGPU setup, and main logic
for everything. Actual blur algorithms and various helpers are under `js/`:

- `separable-blur.js`, `smol-gaussian.js`, `dual-kawase.js`,
  `fast-gaussian.js`, `skia-gaussian.js`, `ryg-blur.js` are what their names suggest, duh.
- `exrloader.js` and `fflate.js`: EXR decoding and decompression,
- `video-export.js`: movie encoding (AV1 and H.264, a tiny MP4 writer),
- `benchmark.js`: benchmark timing, SVG chart generation, the works,
- `gpu-helpers.js`: misc. GPU related helpers.

### External code

- `js/skia-gaussian.js` is adaptation of Google Skia gaussian blur code
  from `SkImageFilterTypes.cpp` and `SkBlurEngine.cpp` (2026 Sept, rev `da51f0d60e`).
  Original is Copyright Google LLC, BSD-3-Clause license.
- `js/fast-gaussian.js` is a port of Blender compositor blur node "Fast Gaussian" related
  code, as it was in 2026 Sept. Original is Copyright 2024 Blender Authors,
  GPL-2.0-or-later license.
- `js/exrloader.js` is an EXR loader adapted from three.js
  ([EXRLoader.js](https://github.com/mrdoob/three.js/blob/65bfbd8e51db/examples/jsm/loaders/EXRLoader.js)),
  incorporating some code from [tinyexr](https://github.com/syoyo/tinyexr) and OpenEXR.
- `js/fflate.js` is the bundled [fflate](https://101arrowz.github.io/fflate/)
  compression library, used for EXR decompression.

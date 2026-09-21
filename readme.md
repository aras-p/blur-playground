# Blur Playground

JavaScript/WebGPU implementation of several image blurring algorithms. Runs in the browser
-- open `html/index.html` (requires WebGPU capable browser).

In order to load provided sample image files, just opening the HTML page
in a browser won't work. Easiest is then to run `python3 -m http.server 8000`
and go to `http://localhost:8000/html/index.html`.

### Blur Algorithms

- **Box**, **Tent**, **Gaussian** - separable blurs
  implemented with the same shader, just different convolution
  kernel shapes. Box and Tent are not great blurs; just here
  because they were easy to do.
- **Dual Kawase** - multi-pass downsample/upsample pyramid blur, from
  Marius Bjørge, [Bandwidth-Efficient Rendering](https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf) (SIGGRAPH 2015),
  see also explanation in [this blog post](https://blog.frost.kiwi/dual-kawase/#dual-kawase-blur).
  This is very fast for large blurs. The original algorithm blurs the same amount
  horizontally & vertically and only allows very discrete
  amounts of blur (more or less "powers of two" radii), so to achieve arbitrary radius there's
  an extra step that blends between two discrete blur amounts, similar to how [obs-composite-blur](https://github.com/FiniteSingularity/obs-composite-blur) does it.
- **Split Kawase** - Similar to Dual Kawase, but with separate
  horizontal/vertical passes for different X/Y radii; has configurable sample weights.
  It looks a bit shit, TBH.

### Dual Kawase interpolation

Dual Kawase is the main focus of this playground. The basic algorithm works well;
the implementation extends it in two ways:

- **Arbitrary blur sizes:** between discrete blur steps, linearly blend the
  previous and next blur amounts using the fractional position between steps,
  similar to the OBS blur plugin linked above. Radii are not rounded before
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

Split Kawase explores a separable version of Dual Kawase, using 1D horizontal
and vertical filters. Its visual quality is currently unsatisfactory.

Test inputs are available in the `exr/` folder.

### Features

- Drag & drop or browse for images (PNG/JPG/EXR),
- Adjustable blur radius (X/Y can be locked or independent),
- Split Kawase weight controls.

### External code

- `html/exrloader.js` is EXR file loader, adapted from three.js
  ([file link](https://github.com/mrdoob/three.js/blob/65bfbd8e51db/examples/jsm/loaders/EXRLoader.js))
  which itself is adapted from [tinyexr](https://github.com/syoyo/tinyexr).
- `html/fflate.js` is gzip/deflate decoder needed for EXR loading,
  from [fflate](https://101arrowz.github.io/fflate/).

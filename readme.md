# Blur Playground

Implementation of several image blurring algorithms. Runs in the browser
-- open `html/index.html` (requires WebGPU capable browser).

### Blur Algorithms

- **Box**, **Tent**, **Gaussian** - separable blurs
  implemented with the same shader, just different convolution
  kernel shapes. Box and Tent are not great blurs; just here
  because they were easy to do.
- **Dual Kawase** - multi-pass downsample/upsample pyramid blur, from
  Marius Bjørge, [Bandwidth-Efficient Rendering](https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf) (SIGGRAPH 2015),
  see also explanation in [this blog post](https://blog.frost.kiwi/dual-kawase/#dual-kawase-blur).
  This is very fast for large blurs, however it can only blur the same amount
  horizontally & vertically. Algorithm from the paper only allows very discrete
  amounts of blur (more or less "powers of two" radii), so to achieve arbitrary radius there's
  an extra step that blends between two discrete blur amounts, similar to how [obs-composite-blur](https://github.com/FiniteSingularity/obs-composite-blur) does it.
- **Split Kawase** - Similar to Dual Kawase, but with separate
  horizontal/vertical passes for different X/Y radii; has configurable sample weights.
  It looks a bit shit, TBH.

### Features

- Drag & drop or browse for images (PNG/JPG/EXR),
- Adjustable blur radius (X/Y can be locked or independent),
- RMSE calculation comparing blur output to Gaussian reference,
- Split Kawase weight controls.

### External code

- `html/exrloader.js` is EXR file loader, adapted from three.js
  ([file link](https://github.com/mrdoob/three.js/blob/65bfbd8e51db/examples/jsm/loaders/EXRLoader.js))
  which itself is adapted from [tinyexr](https://github.com/syoyo/tinyexr).
- `html/fflate.js` is gzip/deflate decoder needed for EXR loading,
  from [fflate](https://101arrowz.github.io/fflate/).

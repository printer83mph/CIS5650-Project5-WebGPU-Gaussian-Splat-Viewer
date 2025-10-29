# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

- Thomas Shaw
- Tested on: **Google Chrome 139.0** on
  Windows 11, Ryzen 7 5700x @ 4.67GHz, 32GB, RTX 2070 8GB

### [Live Demo](https://printer83mph.github.io/CIS5650-Project5-WebGPU-Gaussian-Splat-Viewer/)

### Demo Video/GIF

<img src="docs/video/bonsai.webp" width="512px" />

### Analysis

- Comparing our point cloud renderer to our gaussian splat renderer, we find a fairly large degredation in performance. This makes sense — our point cloud renderer only has to draw points, while the gaussian pipeline must run depth sorting, covariance computation, and overdraw many pixels from each overlapping splat quad.

- Updating the workgroup size leads to strange behavior. This may be due to my implementation, but it seems that any workgroup size outside of 256 will lead to depth sorting artifacts and reduced performance. It may be that smaller workgroups greatly increase the required depth of the parallel radix sort.
  - 64: 30-40fps, rendering artifacts
  - 128: 50-100fps, rendering artifacts
  - 256: 80-120fps
  - \> 256: invalid size for webgpu

- View-frustum culling gives a small performance boost only when looking away from some of the scene. It seems that a large part of the performance cost is simply drawing all of the quads.

- With a larger number of gaussians, scene load time increases dramatically, and performance degrades quite a bit. However, of course, it is only when facing the model and thus drawing these quads that the performance hit is seen.

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

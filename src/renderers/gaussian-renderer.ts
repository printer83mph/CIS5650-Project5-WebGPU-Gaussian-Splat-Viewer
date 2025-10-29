import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl?raw';
import renderWGSL from '../shaders/gaussian.wgsl?raw';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  updateRenderSettings(device: GPUDevice): void;
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {
  const sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const renderSettingsBuffer = device.createBuffer({
    label: 'render settings',
    size: 4 * 2, // 2x f32
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });
  const renderSettingsData = new Uint32Array(2);
  const updateRenderSettings = (
    dev: GPUDevice,
    settings?: { gaussianScaling?: number; shDeg?: number },
  ) => {
    renderSettingsData.set([
      settings?.gaussianScaling ?? renderSettingsData[0],
      settings?.shDeg ?? renderSettingsData[1],
    ]);
    dev.queue.writeBuffer(renderSettingsBuffer, 0, renderSettingsData);
  };
  updateRenderSettings(device, { gaussianScaling: 1, shDeg: pc.sh_deg });

  const splatByteSize = 4 * 5; // 5x u32
  const splatsBuffer = device.createBuffer({
    label: 'splats',
    size: pc.num_points * splatByteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // BELOW: JUST FOR RENDER PIPELINE

  const indirectDrawBuffer = device.createBuffer({
    label: 'indirect draw',
    size: 4 * 4, // 4x u32
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
  });
  // we do 2 tris => 6 verts, instanced num_points times
  const indirectDrawData = new Uint32Array([6, pc.num_points, 0, 0]);
  device.queue.writeBuffer(indirectDrawBuffer, 0, indirectDrawData);

  // ===============================================
  //    Create Bind Group Layouts
  // ===============================================

  const cameraUniformsBindGroupLayout = device.createBindGroupLayout({
    label: 'camera uniforms layout',
    entries: [
      {
        binding: 0, // CameraUniforms
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
      {
        binding: 1, // RenderSettings
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const gaussianBindGroupLayout = device.createBindGroupLayout({
    label: 'gaussians and splats layout',
    entries: [
      {
        binding: 0, // gaussians
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1, // sh coefficients
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2, // splats
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  const sorterBindGroupLayout = device.createBindGroupLayout({
    label: 'sorting bind group layout',
    entries: [...Array(4).keys()].map((idx) => ({
      binding: idx,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    })),
  });

  const splatsBindGroupLayout = device.createBindGroupLayout({
    label: 'splats bind group layout',
    entries: [
      {
        binding: 0, // splats
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1, // splat indices
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
    ],
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  const preprocessPipelineLayout = device.createPipelineLayout({
    label: 'gaussian preprocess layout',
    bindGroupLayouts: [
      cameraUniformsBindGroupLayout,
      gaussianBindGroupLayout,
      sorterBindGroupLayout,
    ],
  });

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: preprocessPipelineLayout,
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const cameraUniformsBindGroup = device.createBindGroup({
    label: 'camera uniforms',
    layout: cameraUniformsBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: renderSettingsBuffer } },
    ],
  });

  const gaussianBindGroup = device.createBindGroup({
    label: 'gaussians and splats',
    layout: gaussianBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splatsBuffer } },
    ],
  });

  const sortBindGroup = device.createBindGroup({
    label: 'sort',
    layout: sorterBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const renderPipelineLayout = device.createPipelineLayout({
    label: 'gaussian render layout',
    bindGroupLayouts: [splatsBindGroupLayout],
  });

  const renderShaderModule = device.createShaderModule({
    label: 'gaussian render module',
    code: renderWGSL,
  });

  const renderPipeline = device.createRenderPipeline({
    label: 'gaussian render pipeline',
    layout: renderPipelineLayout,
    vertex: {
      module: renderShaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: 'fs_main',
      targets: [
        {
          format: presentation_format,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
      frontFace: 'ccw',
    },
  });

  const splatsBindGroup = device.createBindGroup({
    label: 'splats bind group',
    layout: splatsBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: splatsBuffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const preprocess = (encoder: GPUCommandEncoder) => {
    const computePass = encoder.beginComputePass({
      label: 'gaussian preprocess',
    });
    computePass.setPipeline(preprocess_pipeline);
    computePass.setBindGroup(0, cameraUniformsBindGroup);
    computePass.setBindGroup(1, gaussianBindGroup);
    computePass.setBindGroup(2, sortBindGroup);

    const workgroupCount = Math.ceil(pc.num_points / C.histogram_wg_size);
    computePass.dispatchWorkgroups(workgroupCount);

    computePass.end();
  };

  const render = (encoder: GPUCommandEncoder, view: GPUTextureView) => {
    const renderPass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      // no need for depth buffer, we sort everything anyway
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, splatsBindGroup);

    renderPass.drawIndirect(indirectDrawBuffer, 0);

    renderPass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // run preprocess compute pipeline
      sorter.reset(encoder);
      preprocess(encoder);

      // sorter.sort(encoder);
      // TODO: feed sorter output to render pipeline

      // run indirect rendering pipeline
      render(encoder, texture_view);
    },
    camera_buffer,
    updateRenderSettings,
  };
}

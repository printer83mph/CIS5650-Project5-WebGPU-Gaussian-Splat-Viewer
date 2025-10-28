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

  const nullBuffer = device.createBuffer({
    label: 'null buffer',
    size: 4,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const nulling_data = new Uint32Array([0]);
  device.queue.writeBuffer(nullBuffer, 0, nulling_data);

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

  const splatByteSize = 4 * 1; // just position for now
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
    label: 'sort bind group layout',
    entries: Array(4).map((_, idx) => ({
      binding: idx,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    })),
  });

  const preprocessPipelineLayout = device.createPipelineLayout({
    label: 'preprocess layout',
    bindGroupLayouts: [
      cameraUniformsBindGroupLayout,
      gaussianBindGroupLayout,
      sorterBindGroupLayout,
    ],
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
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
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
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

  const render = (encoder: GPUCommandEncoder) => {
    // TODO
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);
      preprocess(encoder);
    },
    camera_buffer,
    updateRenderSettings,
  };
}

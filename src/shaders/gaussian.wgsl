struct Splat {
    position: u32,
    color: array<u32, 2>, // 3x f16 rgb, 1x f16 opacity
    conic: array<u32, 2>, // 3x f16 cov, 1x f16 radius
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> sorted_splat_indices: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

const VERT_OFFSETS = array(
    vec2f(-.5, .5),
    vec2f(.5, .5),
    vec2f(-.5, -.5),
    vec2f(.5, -.5),
    vec2f(-.5, -.5),
    vec2f(.5, .5),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vert_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    // TODO
    // let splat_idx = sorted_splat_indices[instance_idx];

    // let splat = splats[splat_idx];
    let splat = splats[instance_idx];
    let center_pos = unpack2x16float(splat.position);

    let vertPosOffset: vec2f = VERT_OFFSETS[vert_idx];
    let vertPos = center_pos + vertPosOffset * 0.01;

    var out: VertexOutput;

    out.position = vec4<f32>(vertPos, 0., 1.);
    out.color = vec4f(f32(3000) / f32(sorted_splat_indices[instance_idx]), 1., 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

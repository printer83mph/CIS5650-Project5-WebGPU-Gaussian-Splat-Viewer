struct Splat {
    position: u32, // 2x f16
    color: array<u32, 2>, // 3x f16 rgb, 1x f16 opacity
    conic: array<u32, 2>, // 3x f16 cov, 1x f16 radius
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> splat_indices: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) splat_center: vec2<f32>,
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
    let splat_idx = splat_indices[instance_idx];
    let splat = splats[splat_idx];

    var out: VertexOutput;

    let center_pos = unpack2x16float(splat.position);
    out.splat_center = center_pos;

    let vertPosOffset: vec2f = VERT_OFFSETS[vert_idx];
    let vertPos = center_pos + vertPosOffset * 0.01;
    out.position = vec4<f32>(vertPos, 0., 1.);

    let color_rg = unpack2x16float(splat.color[0]);
    let color_b_opacity = unpack2x16float(splat.color[1]);
    out.color = vec4f(color_rg, color_b_opacity);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4f(in.color.rgb, 1.);
}

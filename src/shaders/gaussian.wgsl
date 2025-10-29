struct Splat {
    position: u32, // 2x f16
    color: array<u32, 2>, // 3x f16 rgb, 1x f16 opacity
    conic: array<u32, 2>, // 3x f16 cov, 1x f16 radius
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> splat_indices: array<u32>;

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
// can also grab render settings if needed

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) splat_center: vec2<f32>,
    @location(2) conic: vec3<f32>,
};

const VERT_OFFSETS = array(
    vec2f(-1., 1.),
    vec2f(1., 1.),
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., -1.),
    vec2f(1., 1.),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vert_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let splat_idx = splat_indices[instance_idx];
    let splat = splats[splat_idx];

    var out: VertexOutput;

    // unpack pos
    let center_pos = unpack2x16float(splat.position);
    // give it to fragment shader in pixel space
    out.splat_center = (center_pos * vec2f(0.5, -0.5) + 0.5) * camera.viewport;

    // unpack color and opacity
    out.color = vec4f(unpack2x16float(splat.color[0]), unpack2x16float(splat.color[1]));

    // unpack conic and radius
    let conic_xy = unpack2x16float(splat.conic[0]);
    let conic_z_radius = unpack2x16float(splat.conic[1]);
    out.conic = vec3f(conic_xy, conic_z_radius.x);
    let radius = conic_z_radius.y;

    // set vert pos based on radius
    let vert_pos_offset: vec2f = VERT_OFFSETS[vert_idx] * radius;
    let vert_pos = center_pos + vert_pos_offset / camera.viewport;
    out.position = vec4<f32>(vert_pos, 0., 1.);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // convert to NDC, get distance
    let d = in.position.xy - in.splat_center;
    
    // use insane conic matrix
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    if power > 0.0 {
        discard;
    }
    let alpha = min(0.99, in.color.a * exp(power));
    
    return vec4f(in.color.rgb, alpha);
}

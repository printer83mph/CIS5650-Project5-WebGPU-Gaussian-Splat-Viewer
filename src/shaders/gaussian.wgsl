struct Splat {
    position: u32
    // TODO: probably more info, sync with definition from preprocess
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

@vertex
fn vs_main(
    @builtin(vertex_index) vert_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass
    var out: VertexOutput;
    out.position = vec4<f32>(1., 1., 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}

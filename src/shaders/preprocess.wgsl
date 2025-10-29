const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    position: u32, // 2x f16
    color: array<u32, 2>, // 3x f16 rgb, 1x f16 opacity
    conic: array<u32, 2>, // 3x f16 cov, 1x f16 radius
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

@group(1) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read> sh_coefficients: array<u32>;
@group(1) @binding(2) var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1) var<storage, read_write> sort_depths: array<u32>;
@group(2) @binding(2) var<storage, read_write> sort_indices: array<u32>;
@group(2) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let channel = c_idx & 1u;
    let start_idx = splat_idx * 24 + (c_idx / 2) * 3 + channel;

    let col_01 = unpack2x16float(sh_coefficients[start_idx]);
    let col_23 = unpack2x16float(sh_coefficients[start_idx + 1]);

    if channel == 0u {
        return vec3f(col_01.xy, col_23.x);
    } else {
        return vec3f(col_01.y, col_23.xy);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) wgs: vec3<u32>
) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[idx];

    // unpack position and opacity
    let pos_xy = unpack2x16float(gaussian.pos_opacity[0]);
    let pos_z_opacity = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4f(pos_xy, pos_z_opacity.x, 1.);

    var pos_view = camera.view * pos;
    var pos_ndc = camera.proj * pos_view;
    pos_ndc /= pos_ndc.w;

    // simple frustum culling
    if any(abs(pos_ndc.xy) > vec2f(1.2)) || pos_ndc.z < 0. {
        return;
    }

    // compute color and opacity
    let camera_to_center = normalize(pos.xyz + camera.view[3].xyz);
    let color = computeColorFromSH(camera_to_center, idx, u32(render_settings.sh_deg));
    let opacity = 1.0 / (1.0 + exp(-pos_z_opacity.y));

    // unpack rot and scale
    let rot = vec4f(unpack2x16float(gaussian.rot[0]), unpack2x16float(gaussian.rot[1]));
    let scale = vec3f(unpack2x16float(gaussian.scale[0]), unpack2x16float(gaussian.scale[1]).x);

    // compute conic values
    let R = mat3x3f(
        1. - 2. * (rot.y * rot.y + rot.z * rot.z), 2. * (rot.x * rot.y - rot.r * rot.z), 2. * (rot.x * rot.z + rot.r * rot.y),
        2. * (rot.x * rot.y + rot.r * rot.z), 1. - 2. * (rot.x * rot.x + rot.z * rot.z), 2. * (rot.y * rot.z - rot.r * rot.x),
        2. * (rot.x * rot.z - rot.r * rot.y), 2. * (rot.y * rot.z + rot.r * rot.x), 1. - 2. * (rot.x * rot.x + rot.y * rot.y)
    );
    var S = mat3x3f(
        render_settings.gaussian_scaling * scale.x, 0., 0.,
        0., render_settings.gaussian_scaling * scale.y, 0.,
        0., 0., render_settings.gaussian_scaling * scale.z,
    );

    let cov3D = transpose(S * R) * S * R;

    let t = pos_view.xyz;
    let J = mat3x3f(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );

    let W = transpose(mat3x3f(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    ));

    let T = W * J;

    let Vrk = mat3x3f(
        cov3D[0][0], cov3D[0][1], cov3D[0][2],
        cov3D[0][1], cov3D[1][1], cov3D[1][2],
        cov3D[0][2], cov3D[1][2], cov3D[2][2]
    );

    var cov2D = transpose(T) * transpose(Vrk) * T;

    // this addition is for stability apparently?
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;

    // compute conic and radius
    let cov = vec3f(cov2D[0][0], cov2D[0][1], cov2D[1][1]);
    let det = cov.x * cov.z - cov.y * cov.y;
    if det == 0.0 {
        return;
    }

    // compute radius
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det)); // The author is not too sure what 0.1 serves here :o
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // compute conic
    let inv_det = 1.0 / det;
    let conic = vec3f(cov.z * inv_det, -cov.y * inv_det, cov.x * inv_det);

    let splat_idx = atomicAdd(&sort_infos.keys_size, 1u);
    splats[splat_idx].position = pack2x16float(pos_ndc.xy);
    splats[splat_idx].color[0] = pack2x16float(color.rg);
    splats[splat_idx].color[1] = pack2x16float(vec2f(color.b, opacity));
    splats[splat_idx].conic[0] = pack2x16float(conic.xy);
    splats[splat_idx].conic[1] = pack2x16float(vec2f(conic.z, radius));

    // update sorting info!
    sort_indices[splat_idx] = splat_idx;

    let depth = -pos_view.z;
    let depth_bits = bitcast<u32>(depth);

    // Flip all bits if negative (sign bit set), otherwise flip just the sign bit
    let sortable_depth = select(
        depth_bits ^ 0x80000000u,  // Positive: flip sign bit
        ~depth_bits,               // Negative: flip all bits
        (depth_bits & 0x80000000u) != 0u
    );
    sort_depths[splat_idx] = sortable_depth;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if splat_idx % keys_per_dispatch == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    } // how the hell does this work
}

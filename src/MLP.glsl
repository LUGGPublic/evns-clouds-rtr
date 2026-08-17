#ifndef MLP_GLSL
#define MLP_GLSL

#extension GL_EXT_debug_printf : enable
#define printf debugPrintfEXT

#define COOP_VECTOR_TYPE gl_ComponentTypeFloat16NV
#extension GL_NV_cooperative_vector : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(buffer_reference, scalar) readonly buffer MLPParam
{
    float16_t values[];
};

layout(buffer_reference, scalar) readonly buffer MLPOffset
{
    uint values[];
};

layout(constant_id = 0) const int input_count = 14;
layout(constant_id = 1) const int output_count = 7;
layout(constant_id = 2) const int layer_count = 4;
layout(constant_id = 3) const int mlp_dim = 32;
layout(constant_id = 4) const int version = 0;
layout(constant_id = 5) const int triplane_resolution = 512;
layout(constant_id = 6) const int triplane_features = 8;

struct MLP {
    uvec2 params;
    uvec2 weightOffsets;
    uvec2 biasOffsets;
};

layout(set = 4, binding = 1) uniform sampler2D triplaneXY[2];
layout(set = 4, binding = 2) uniform sampler2D triplaneXZ[2];
layout(set = 4, binding = 3) uniform sampler2D triplaneYZ[2];

/* Calculate offset from address */
uvec2 addr_offset(uvec2 addr, uint offset)
{
    uint carry;
    addr.x = uaddCarry(addr.x, offset, carry);
    addr.y += carry;
    return addr;
}

vec2 convertTriplaneUV(vec2 p, float mipmap) {
    vec2 size = vec2(textureSize(triplaneXY[0], int(mipmap)));
    vec2 uv = p * ((size - 1) / size) + 0.5 / size;
    return uv;
}

void triplane_encoding(vec3 pos, float mipmap, out coopvecNV<float16_t, input_count> enc)
{
    vec4 xy0 = textureLod(triplaneXY[0], convertTriplaneUV(pos.xy, mipmap), mipmap);
    vec4 xy1 = textureLod(triplaneXY[1], convertTriplaneUV(pos.xy, mipmap), mipmap);
    vec4 xz0 = textureLod(triplaneXZ[0], convertTriplaneUV(pos.xz, mipmap), mipmap);
    vec4 xz1 = textureLod(triplaneXZ[1], convertTriplaneUV(pos.xz, mipmap), mipmap);
    vec4 yz0 = textureLod(triplaneYZ[0], convertTriplaneUV(pos.yz, mipmap), mipmap);
    vec4 yz1 = textureLod(triplaneYZ[1], convertTriplaneUV(pos.yz, mipmap), mipmap);
    vec4 f0 = xy0 + xz0 + yz0;
    vec4 f1 = xy1 + xz1 + yz1;
    enc[0] = float16_t(f0.x);
    enc[1] = float16_t(f0.y);
    enc[2] = float16_t(f0.z);
    enc[3] = float16_t(f0.w);
    enc[4] = float16_t(f1.x);
    enc[5] = float16_t(f1.y);
    enc[6] = float16_t(f1.z);
    enc[7] = float16_t(f1.w);
}

void mlp_forward(MLP mlp, float triplaneMipmap, in float inputs[9], out float outputs[7])
{
    MLPParam params = MLPParam(mlp.params);
    MLPOffset weightOffsets = MLPOffset(mlp.weightOffsets);
    MLPOffset biasOffsets = MLPOffset(mlp.biasOffsets);

    coopvecNV<float16_t, input_count> input_vec;
    coopvecNV<float16_t, mlp_dim> mlp_vec;
    coopvecNV<float16_t, output_count> output_vec;

    vec3 pos = vec3(inputs[0], inputs[1], inputs[2]);
    triplane_encoding(pos, triplaneMipmap, input_vec);       // Triplane encoding
    input_vec[triplane_features + 0] = float16_t(inputs[3]); // V.x
    input_vec[triplane_features + 1] = float16_t(inputs[4]); // V.y
    input_vec[triplane_features + 2] = float16_t(inputs[5]); // V.z
    input_vec[triplane_features + 3] = float16_t(inputs[6]); // Sun Elevation
    input_vec[triplane_features + 4] = float16_t(inputs[7]); // Sun Rotation

    if (input_count == 14) {
        input_vec[triplane_features + 5] = float16_t(inputs[8]); // Thickness
    }

    /* Input layer */
    coopVecMatMulAddNV(mlp_vec, input_vec, COOP_VECTOR_TYPE, params.values, weightOffsets.values[0], COOP_VECTOR_TYPE,
                       params.values, biasOffsets.values[0], gl_ComponentTypeFloat16NV, mlp_dim, input_count,
                       gl_CooperativeVectorMatrixLayoutInferencingOptimalNV, false, 0);

    /* ReLU activation */
    mlp_vec = max(coopvecNV<float16_t, mlp_dim>(float16_t(0)), mlp_vec);

    /* Hidden layers */
    for (int i = 1; i < layer_count - 1; i++) {
        coopVecMatMulAddNV(mlp_vec, mlp_vec, COOP_VECTOR_TYPE, params.values, weightOffsets.values[i], COOP_VECTOR_TYPE,
                           params.values, biasOffsets.values[i], gl_ComponentTypeFloat16NV, mlp_dim, mlp_dim,
                           gl_CooperativeVectorMatrixLayoutInferencingOptimalNV, false, 0);

        /* ReLU activation */
        mlp_vec = max(coopvecNV<float16_t, mlp_dim>(float16_t(0)), mlp_vec);
    }

    /* Output layer */
    coopVecMatMulAddNV(output_vec, mlp_vec, COOP_VECTOR_TYPE, params.values, weightOffsets.values[layer_count - 1],
                       COOP_VECTOR_TYPE, params.values, biasOffsets.values[layer_count - 1], gl_ComponentTypeFloat16NV,
                       output_count, mlp_dim, gl_CooperativeVectorMatrixLayoutInferencingOptimalNV, false, 0);

    /* No activation, deal with it in fragment shader */

    /* Set output */
    outputs[0] = float(output_vec[0]);
    outputs[1] = float(output_vec[1]);
    outputs[2] = float(output_vec[2]);
    outputs[3] = float(output_vec[3]);
    outputs[4] = float(output_vec[4]);
    outputs[5] = float(output_vec[5]);
    outputs[6] = float(output_vec[6]);
}

#endif

#version 460

#extension GL_GOOGLE_include_directive : require

#include "Extensions.glsl"
#include "MLP.glsl"

layout(location = 0) in vec3 inPositionWorld;
layout(location = 1) in vec3 inPositionModelNorm;
//layout(location = 2) in float inViewDepth;

layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform CameraUniformDynamic {
    mat4 view;
    mat4 view_inv;
    mat4 proj;
    mat4 proj_inv;
} camera;

layout(set = 2, binding = 0) uniform MaterialParams {
    vec3 diffuse;
    float shininess;
    vec3 specular;
    float indexOfRefraction;
    vec3 ambient;
    float opacity;
    vec3 emission;
    uint hasTexture;
} materialParams;

layout(set = 2, binding = 1) uniform sampler2D diffuseTexture;
layout(set = 2, binding = 2) uniform sampler2D specularTexture;
layout(set = 2, binding = 3) uniform sampler2D ambientTexture;
layout(set = 2, binding = 4) uniform sampler2D emissionTexture;
layout(set = 2, binding = 5) uniform sampler2D normalTexture;

layout(set = 3, binding = 0) uniform sampler2D environmentMap;

layout(set = 4, binding = 0, scalar) readonly buffer Locations0 {
    uvec2 params;
    uvec2 weightOffsets;
    uvec2 biasOffsets;
} locMlp;

layout(set = 5, binding = 0, rgba16f) uniform image2D inBackPosition;

layout(set = 6, binding = 0) uniform sampler2D inShadowMap;
layout(set = 7, binding = 0) uniform LightUniformDynamic {
    mat4 view;
    mat4 view_inv;
    mat4 proj;
    mat4 proj_inv;
} light;

layout(push_constant) uniform PushConstant {
    vec2 viewPort;
    float sunElevation;
    float sunRotation;
    vec3 minBounds;
	int renderMode;
    vec3 maxBounds;
    float exposure;
    float triplaneMipmap;
} pushConstant;

vec3 linear_to_srgb(vec3 color) {
    vec3 low = 12.92 * color;
    vec3 high = 1.055 * pow(color, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(low, high, step(vec3(0.0031308), color));
}

vec3 apply_exposure(vec3 color) {
    return 1.0 - exp(-color * pushConstant.exposure);
}

float computeWeight(vec3 color, float alpha, float depth) {
    return max(min(1.0, max(max(color.r, color.g), color.b) * alpha), alpha) *
    clamp(0.03 / (1e-5 + pow(depth / 2000, 4.0)), 1e-2, 3e3);
}

void main() {
    MLP mlp = MLP(locMlp.params, locMlp.weightOffsets, locMlp.biasOffsets);

    vec3 V = normalize(camera.view_inv[3].xyz - inPositionWorld);
    V = V.xzy; // Swap Y-Z to fit Blender world space coordinate system
    V.y = -V.y; // Flip Y to fit Blender world space coordinate system

    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec3 backPosition = imageLoad(inBackPosition, coord).xyz;
    float thickness = distance(backPosition, inPositionModelNorm);
    thickness = smoothstep(0.0, 0.2, thickness);

    float inputs[9];
    inputs[0] = inPositionModelNorm.x; 
    inputs[1] = inPositionModelNorm.y;
    inputs[2] = inPositionModelNorm.z;
//    inputs[3] = V.x;
//    inputs[4] = V.y;
//    inputs[5] = V.z;
//    inputs[6] = (pushConstant.sunElevation + radians(5.0)) / radians(95.0);
//    inputs[7] = pushConstant.sunRotation / radians(360.0);
//    inputs[8] = thickness;
    inputs[3] = (pushConstant.sunElevation + radians(5.0)) / radians(95.0);
    inputs[4] = (pushConstant.sunRotation + radians(90.0)) / radians(360.0);
    inputs[5] = thickness;

    float outputs[7];
    mlp_forward(mlp, pushConstant.triplaneMipmap, inputs, outputs);

    // Extract output and apply final activation
    vec3 rgb_vis = exp(vec3(outputs[0], outputs[1], outputs[2]) - 3.0);
    vec3 rgb_hid = exp(vec3(outputs[3], outputs[4], outputs[5]) - 3.0);
    float alpha = exp(outputs[6] - 3.0);

    // Do shadow mapping (PCF)
    vec4 posLightScreen = light.proj * light.view * vec4(inPositionWorld, 1.0);
    posLightScreen /= posLightScreen.w;
    posLightScreen = posLightScreen * 0.5 + 0.5;

    vec2 texelSize = 1.0 / textureSize(inShadowMap, 0);
    float bias = 2e-4;
	mat3 weights = mat3(1.5, 2.0, 1.5,
                        2.0, 3.0, 1.5,
                        1.5, 2.0, 1.5);
    float shadowRatio = 0.0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
        vec2 uv = posLightScreen.xy + vec2(i, j) * texelSize;
            float shadowMapDepth = texture(inShadowMap, uv).r;
            shadowRatio += (posLightScreen.z * 2.0 - 1.0) > shadowMapDepth + bias ? 0 : weights[i + 1][j + 1];
        }
    }
    shadowRatio /= 17.0;


    // Pick RGB output depending on shadow map
    vec3 rgb = mix(rgb_hid, rgb_vis, shadowRatio);
//    rgb = rgb_vis;

    if (pushConstant.renderMode == 0) {
        fragColor = vec4(rgb, alpha);
    } else if (pushConstant.renderMode == 1) {
        fragColor = vec4(inPositionModelNorm, 1.0);
    } else if (pushConstant.renderMode == 2) {
        fragColor = vec4(backPosition, 1.0);
    } else if (pushConstant.renderMode == 3) {
        fragColor = vec4(V * 0.5 + 0.5, 1.0);
    } else if (pushConstant.renderMode == 4) {
        fragColor = vec4(vec3(thickness), 1.0);
    }
}

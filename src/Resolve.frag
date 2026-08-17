#version 460

layout(location = 0) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;

layout(set = 1, binding = 0, rgba16f) uniform image2D inCloudRenderTarget;

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

void main() {
    vec4 cloud = imageLoad(inCloudRenderTarget, ivec2(gl_FragCoord.xy));
    if (pushConstant.renderMode != 0) {
        fragColor = cloud;
    } else {
        fragColor = vec4(linear_to_srgb(apply_exposure(cloud.rgb)), cloud.a);
    }
}

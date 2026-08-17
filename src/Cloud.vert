#version 460

layout(set = 0, binding = 0) uniform CameraUniformDynamic {
    mat4 view;
    mat4 view_inv;
    mat4 proj;
    mat4 proj_inv;
} camera;

layout(set = 1, binding = 0) uniform MeshUniformDynamic {
    mat4 model;
} mesh;

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

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexTextureCoord;
layout(location = 3) in vec3 vertexTangent;
layout(location = 4) in vec3 vertexBinormal;

layout(location = 0) out vec3 outPositionWorld;
layout(location = 1) out vec3 outPositionModelNorm;

void main() {
    // Position in world space
    outPositionWorld = (mesh.model * vec4(vertexPosition, 1.0)).xyz;

    // Position for MLP in model space, normalized to bounding box in Blender space
    outPositionModelNorm = vertexPosition.xzy; // Swap Y-Z to fit Blender world space coordinate system
    outPositionModelNorm.y = -outPositionModelNorm.y; // Flip Y to fit Blender world space coordinate system
    outPositionModelNorm = (outPositionModelNorm - pushConstant.minBounds) / (pushConstant.maxBounds - pushConstant.minBounds);

    gl_Position = camera.proj * camera.view * mesh.model * vec4(vertexPosition, 1.0);
}

#version 460

layout(location = 0) in vec3 inPositionWorld;
layout(location = 1) in vec3 inPositionModelNorm;

layout(location = 0) out vec4 outPosition;

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

void main() {
    outPosition = vec4(inPositionModelNorm, 1.0);
}

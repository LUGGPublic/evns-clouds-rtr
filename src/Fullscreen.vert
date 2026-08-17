#version 460

layout(location = 0) out vec2 outTexCoord;

layout(set = 0, binding = 0) uniform CameraUniformDynamic {
    mat4 view;
    mat4 view_inv;
    mat4 proj;
    mat4 proj_inv;
} camera;

void main() {
	outTexCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outTexCoord * 2.0 - 1.0, 0.0, 1.0);
}

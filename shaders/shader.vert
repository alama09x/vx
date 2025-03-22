#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(binding = 0) uniform Camera {
    mat4 view;
    mat4 projection;
} camera;

void main() {
   gl_Position = camera.projection * camera.view * vec4(in_position, 1.0);
   out_color = in_color;
}
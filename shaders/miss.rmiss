#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 hit_value;

void main() {
    hit_value = vec3(0.1, 0.1, 0.2);
}
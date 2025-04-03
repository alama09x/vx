#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 hit_value;
hitAttributeEXT vec3 attribs;

void main() {
    // Vertex color from hit
    hit_value = attribs;
}
#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 hit_value;
hitAttributeEXT vec2 attribs;

void main() {
    // Vertex color from hit
    // hit_value = vec3(attribs, 0.0);
    hit_value = vec3(1.0, 0.0, 0.0);
}
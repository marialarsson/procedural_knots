#version 330
#extension GL_ARB_explicit_attrib_location : require
#extension GL_ARB_explicit_uniform_location : require

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 transform;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float rmin;
uniform float hmax;

out vec3 outTexCoords;
out vec3 normal;
out vec3 fragPos;
out float zrat;

void main()
{
    gl_Position =  view * model * vec4(in_position, 1.0);
    zrat = hmax/rmin;
    outTexCoords = vec3( in_position.x, in_position.y, in_position.z);
    fragPos = vec3( model * vec4(in_position, 1.0));
    normal = mat3(transpose(inverse(model))) * in_normal;
}

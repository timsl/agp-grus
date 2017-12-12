#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 M;
layout(location = 2) in float type;

uniform mat4 VP;

out float c_type;

void main(){
    gl_Position = VP * (M + vec4(pos.x, pos.y, pos.z, 1.0));
    c_type = type;
}

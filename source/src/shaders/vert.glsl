#version 450 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 M;
layout(location = 2) in uint type;

uniform mat4 VP;

flat out uint c_type;

void main(){
    gl_Position = VP * (M + vec4(pos.x, pos.y, pos.z, 1.0));
    c_type = type;
}

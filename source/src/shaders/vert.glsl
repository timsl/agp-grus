#version 330 core
layout(location = 0) in vec3 pos;
uniform mat4 P;

void main(){
        gl_Position = P * vec4(pos.x, pos.y, pos.z, 1.0);
}

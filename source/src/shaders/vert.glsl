#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in mat4 mv;
uniform mat4 P;

void main(){
        gl_Position = mv * vec4(pos.x, pos.y, pos.z, 1.0);
}

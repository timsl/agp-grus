#version 330 core

layout(location = 0) in vec3 pos;
layout(location = 1) in mat4 model_view;
layout(location = 5) in float in_type;

uniform mat4 P;

out float c_type;

void main(){
    gl_Position = P * model_view * vec4(pos.x, pos.y, pos.z, 1.0);
    c_type = in_type;
}

#version 450 core

flat in uint c_type;

uniform vec4 uColor[4];

out vec4 FragColor;

void main()
{
        FragColor = uColor[c_type];
}

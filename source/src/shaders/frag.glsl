#version 330 core

in float c_type;

uniform vec4 uColor[4];

out vec4 FragColor;

void main()
{
	FragColor = uColor[int(c_type)];
}

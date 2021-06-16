#version 430

in vec2 uv;

uniform sampler2D textureMap;

out vec4 outColor;

void main()
{
    outColor = texture(textureMap, uv);
}
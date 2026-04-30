#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform float pointRadius;
uniform float screenHeight;

void main()
{
    vec4 viewPos = viewMatrix * vec4(aPos, 1.0);
    float pointSize = screenHeight * (projectionMatrix[1][1] * pointRadius) / -viewPos.z;
    gl_Position = projectionMatrix * viewPos;
    gl_PointSize = pointSize;
}

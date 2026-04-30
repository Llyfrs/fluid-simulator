#version 330 core

out vec4 FragColor;

uniform vec3 lightDir; // A simple directional light

void main()
{
    // gl_PointCoord gives us the 2D coordinate inside the point sprite, from [0,1]
    // We remap it to [-1, 1] to make it a unit circle
    vec2 p = gl_PointCoord * 2.0 - 1.0;

    // If the pixel is outside the circle, discard it.
    // This makes our square shape round.
    if (dot(p, p) > 1.0) {
        discard;
    }

    // Calculate the 3D normal of the sphere at this pixel
    // This gives it the illusion of volume.
    vec3 normal;
    normal.xy = p;
    normal.z = sqrt(1.0 - dot(p, p));

    // Simple diffuse lighting
    float diffuse = max(0.3, dot(normal, normalize(lightDir)));

    vec3 sphereColor = vec3(0.28, 0.90, 0.98); // Blue fluid
    FragColor = vec4(sphereColor * diffuse, 1.0);
}
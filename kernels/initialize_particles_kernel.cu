#include <cuda_runtime.h> // For float3, int3 which are useful host/device types
#include "cuda_kernels.cuh"

/**
 * @brief A simple stateless pseudo-random number generator for the GPU.
 *        Produces a deterministic float value between [0.0, 1.0] based on two seeds.
 *
 * @param seed1 A seed value, can be based on particle index.
 * @param seed2 Another seed value, can be used for different dimensions (x, y, z).
 * @return A float value in the range [0.0, 1.0].
 */
__device__ float pseudo_random(unsigned int seed1, unsigned int seed2) {
    // A common integer hash function (Thomas Wang's 32-bit integer hash)
    unsigned int a = seed1;
    unsigned int b = seed2;
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    b = (b ^ 61) ^ (b >> 16);
    b = b + (b << 3);
    b = b ^ (b >> 4);
    b = b * 0x27d4eb2d;
    b = b ^ (b >> 15);

    unsigned int combined = a ^ b;

    // Convert the integer hash to a float in [0, 1]
    return (float)(combined & 0xFFFFFF) / 16777215.0f;
}


/**
 * @brief Initializes particles in a structured, ordered grid within a specified
 *        rectangular volume, adding a small random jitter to break up the
 *        perfect alignment. Velocities are initialized to zero.
 *
 * @param d_ps                  Pointer to the particle system on the device.
 * @param min_bounds            The (x, y, z) world coordinates of the corner of the fluid box.
 * @param particles_per_dim     The number of particles to place along each axis (x, y, z).
 * @param particle_spacing      The distance between adjacent particles in world units.
 */
__global__ void initialize_particles_kernel(
        ParticleSystem* d_ps,
        float3 min_bounds,
        int3 particles_per_dim,
        float particle_spacing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_ps->numParticles) return;

    // --- 1. Un-flatten the 1D particle index into a 3D grid index (p_i, p_j, p_k) ---
    // This gives us the particle's coordinate within the particle grid structure.
    int p_i = idx % particles_per_dim.x;
    int temp_idx = idx / particles_per_dim.x;
    int p_j = temp_idx % particles_per_dim.y;
    int p_k = idx / (particles_per_dim.x * particles_per_dim.y);

    // --- 2. Calculate the particle's base world position (the perfect grid point) ---
    float base_x = min_bounds.x + p_i * particle_spacing;
    float base_y = min_bounds.y + p_j * particle_spacing;
    float base_z = min_bounds.z + p_k * particle_spacing;

    // --- 3. Calculate a small random jitter ---
    // The jitter should be smaller than the particle spacing to maintain uniform density.
    // Jittering by +/- 25% of the spacing is a safe choice.
    float jitter_scale = 0.25f; // Jitter from [-0.25, 0.25] of spacing
    float jitter_dist = particle_spacing * jitter_scale;

    // Use pseudo_random to get a value in [-1, 1] and scale it
    float jitter_x = (pseudo_random(idx, 0u) * 2.0f - 1.0f) * jitter_dist;
    float jitter_y = (pseudo_random(idx, 1u) * 2.0f - 1.0f) * jitter_dist;
    float jitter_z = (pseudo_random(idx, 2u) * 2.0f - 1.0f) * jitter_dist;

    // --- 4. Calculate the final position and store it ---
    d_ps->x[idx] = base_x + jitter_x;
    d_ps->y[idx] = base_y + jitter_y;
    d_ps->z[idx] = base_z + jitter_z;

    // --- 5. Initialize velocities to zero for a fluid starting at rest ---
    d_ps->vx[idx] = 0.0f;
    d_ps->vy[idx] = 0.0f;
    d_ps->vz[idx] = 0.0f;
}
#include "cuda_kernels.cuh"

/**
 * @brief Helper function to reflect a velocity vector across a surface normal.
 *
 * @param v The incoming velocity vector.
 * @param n The surface normal vector (must be normalized).
 * @return The reflected velocity vector.
 */
__device__ inline float3 reflect_velocity(const float3& v, const float3& n) {
    // Implements the reflection formula: v_out = v_in - 2 * dot(v_in, n) * n
    float dot_product = v.x * n.x + v.y * n.y + v.z * n.z;
    return make_float3(
            v.x - 2.0f * dot_product * n.x,
            v.y - 2.0f * dot_product * n.y,
            v.z - 2.0f * dot_product * n.z
    );
}


/**
 * @brief Advects particles and then clamps their final position to stay within the
 *        non-solid region of the grid domain. This is a direct implementation of
 *        the "ClampToNonSolidCells" strategy.
 *
 * @param d_ps Pointer to the particle system on the device.
 * @param d_fg Pointer to the fluid grid on the device.
 * @param dt   The simulation time step.
 */
__global__ void update_particle_positions_kernel(ParticleSystem* d_ps, FluidGrid* d_fg, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_ps->numParticles) {
        return;
    }


    // --- 1. Predict new position (Advection step) ---
    float3 pos;
    pos.x = d_ps->x[idx] + d_ps->vx[idx] * dt;
    pos.y = d_ps->y[idx] + d_ps->vy[idx] * dt;
    pos.z = d_ps->z[idx] + d_ps->vz[idx] * dt;


    // --- 2. Define the valid (non-solid) domain boundaries ---
    // The "cushion" is one grid cell width, ensuring particles stay out of the solid boundary layer.
    const float cushion = d_fg->grid_spacing;

    // Minimum allowed coordinates
    const float3 domain_min = make_float3(
            d_fg->originX + cushion,
            d_fg->originY + cushion,
            d_fg->originZ + cushion
    );

    // Maximum allowed coordinates
    const float3 domain_max = make_float3(
            d_fg->originX + (d_fg->nx - 1) * d_fg->grid_spacing,
            d_fg->originY + (d_fg->ny - 1) * d_fg->grid_spacing ,
            d_fg->originZ + (d_fg->nz - 1) * d_fg->grid_spacing
    );

    // --- 3. Clamp the position and modify velocity upon collision ---
    // Check X-axis
    if (pos.x < domain_min.x) {
        pos.x = domain_min.x; // Ensure we stay out of the solid boundary layer
    } else if (pos.x > domain_max.x) {
        pos.x = domain_max.x; // Ensure we stay out of the solid boundary layer
    }

    // Check Y-axis
    if (pos.y < domain_min.y) {
        pos.y = domain_min.y; // Ensure we stay out of the solid boundary layer
    } else if (pos.y > domain_max.y) {
        pos.y = domain_max.y; // Ensure we stay out of the solid boundary layer
    }

    // Check Z-axis
    if (pos.z < domain_min.z) {
        pos.z = domain_min.z; // Ensure we stay out of the solid boundary layer
    } else if (pos.z > domain_max.z) {
        pos.z = domain_max.z; // Ensure we stay out of the solid boundary layer
    }

    // --- 4. Store final, corrected state ---
    d_ps->x[idx] = pos.x;
    d_ps->y[idx] = pos.y;
    d_ps->z[idx] = pos.z;
}



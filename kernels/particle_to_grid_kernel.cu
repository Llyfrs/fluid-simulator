#include "cuda_kernels.cuh"

// Before optimizing : particle_to_grid executed in 1.95 ms




/**
 * @brief Generic helper function to transfer a single particle's velocity component
 *        to the appropriate staggered grid (U, V, or W).
 *
 * @param type          The type of grid to transfer to (U_GRID, V_GRID, or W_GRID).
 * @param gx, gy, gz    The particle's position in cell-centered grid coordinates.
 * @param particle_vel  The particle's velocity component (vx, vy, or vz) to be transferred.
 * @param d_fg          Pointer to the main fluid grid struct on the device.
 */
template<GridType type>
__device__ void transfer_to_staggered_grid(
        float gx, float gy, float gz,
        float particle_vel,
        FluidGrid* d_fg)
{
    // --- 1. Set up parameters based on the grid type (constexpr branching) ---
    float shift_x, shift_y, shift_z;
    int dim_x, dim_y, dim_z;
    float* vel_array;
    float* weight_array;

    const int nx = d_fg->nx;
    const int ny = d_fg->ny;
    const int nz = d_fg->nz;

    if constexpr (type == U_GRID) {
        // U-grid is shifted by (0, -0.5, -0.5) relative to cell centers
        shift_x = 0.0f;  shift_y = -0.5f; shift_z = -0.5f;
        dim_x = nx + 1;  dim_y = ny;      dim_z = nz;
        vel_array = d_fg->u;
        weight_array = d_fg->u_weights;
    }
    else if constexpr (type == V_GRID) {
        // V-grid is shifted by (-0.5, 0, -0.5)
        shift_x = -0.5f; shift_y = 0.0f;  shift_z = -0.5f;
        dim_x = nx;      dim_y = ny + 1;  dim_z = nz;
        vel_array = d_fg->v;
        weight_array = d_fg->v_weights;
    }
    else { // W_GRID
        // W-grid is shifted by (-0.5, -0.5, 0)
        shift_x = -0.5f; shift_y = -0.5f; shift_z = 0.0f;
        dim_x = nx;      dim_y = ny;      dim_z = nz + 1;
        vel_array = d_fg->w;
        weight_array = d_fg->w_weights;
    }

    // --- 2. Compute shifted particle position relative to staggered grid ---
    float shifted_gx = gx + shift_x;
    float shifted_gy = gy + shift_y;
    float shifted_gz = gz + shift_z;

    // Base indices (integer part)
    int base_i = floorf(shifted_gx);
    int base_j = floorf(shifted_gy);
    int base_k = floorf(shifted_gz);

    // Fractional parts for interpolation weights
    float fx = shifted_gx - base_i;
    float fy = shifted_gy - base_j;
    float fz = shifted_gz - base_k;

    // --- 3. Loop over the 8 surrounding grid nodes and update velocity and weights ---
    for (int k_offset = 0; k_offset <= 1; ++k_offset) {
        for (int j_offset = 0; j_offset <= 1; ++j_offset) {
            for (int i_offset = 0; i_offset <= 1; ++i_offset) {

                int node_i = base_i + i_offset;
                int node_j = base_j + j_offset;
                int node_k = base_k + k_offset;

                // Boundary check
                if (node_i >= 0 && node_i < dim_x &&
                    node_j >= 0 && node_j < dim_y &&
                    node_k >= 0 && node_k < dim_z)
                {
                    // Compute trilinear interpolation weight for this node
                    float weight = (i_offset ? fx : 1.0f - fx) *
                                   (j_offset ? fy : 1.0f - fy) *
                                   (k_offset ? fz : 1.0f - fz);

                    // Compute flattened 1D index for this grid node
                    int index;
                    if constexpr (type == U_GRID)
                        index = u_idx(node_i, node_j, node_k, nx, ny, nz);
                    else if constexpr (type == V_GRID)
                        index = v_idx(node_i, node_j, node_k, nx, ny, nz);
                    else
                        index = w_idx(node_i, node_j, node_k, nx, ny, nz);

                    // Atomically add weighted velocity and weight to grid arrays
                    atomicAdd(&vel_array[index], particle_vel * weight);
                    atomicAdd(&weight_array[index], weight);
                }
            }
        }
    }
}


/**
 * @brief Kernel 1: Transfers particle velocities to a staggered grid, summing
 *        both the weighted velocities and the weights themselves. This is the
 *        core "Particle-to-Grid" (P2G) splatting operation.
 */
__global__ void particle_to_grid_kernel(ParticleSystem* d_ps, FluidGrid* d_fg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_ps->numParticles) {
        return;
    }

    // 1. Get particle data
    float px = d_ps->x[idx];
    float py = d_ps->y[idx];
    float pz = d_ps->z[idx];

    float vx = d_ps->vx[idx];
    float vy = d_ps->vy[idx];
    float vz = d_ps->vz[idx];

    // 2. Convert particle's world position to normalized, cell-centered grid coordinates
    const float inv_dx = 1.0f / d_fg->grid_spacing;
    float gx = (px - d_fg->originX) * inv_dx;
    float gy = (py - d_fg->originY) * inv_dx;
    float gz = (pz - d_fg->originZ) * inv_dx;

    // --- NEW RESPONSIBILITY: MARK THE PARTICLE'S CELL AS FLUID ---
    int cell_i = floorf(gx);
    int cell_j = floorf(gy);
    int cell_k = floorf(gz);

    // Boundary check for safety, though particles should be handled separately
    if (cell_i >= 0 && cell_i < d_fg->nx &&
        cell_j >= 0 && cell_j < d_fg->ny &&
        cell_k >= 0 && cell_k < d_fg->nz) {

        int cell_idx = cc_idx(cell_i, cell_j, cell_k, d_fg->nx, d_fg->ny);

        // Mark the cell as containing fluid.
        // NOTE: This is a "benign race condition". Multiple threads might write
        // the same value (FLUID_CELL) to the same location. This is fine and
        // does not require an atomic operation.
        if (d_fg->cellType[cell_idx] != SOLID_CELL) {
            d_fg->cellType[cell_idx] = FLUID_CELL;
        }
    }

    // 3. Call the generic helper for each velocity component
    transfer_to_staggered_grid<U_GRID>(gx, gy, gz, vx, d_fg);
    transfer_to_staggered_grid<V_GRID>(gx, gy, gz, vy, d_fg);
    transfer_to_staggered_grid<W_GRID>(gx, gy, gz, vz, d_fg);
}
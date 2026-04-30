/** @file alternative_p2g_kernels.cu
 * @brief Alternative P2G kernels for staggered grids. Nothing in this file is used.
 *       This file contains the kernels for building the spatial hash grid, but it did not worked so
 *       they were replaced by the kernels in particle_to_grid_kernel.cu.
 */

#include "cuda_kernels.cuh"



__global__ void build_spatial_hash_kernel(
        ParticleSystem* d_ps, FluidGrid* d_fg,
        int* d_cell_heads, int* d_particle_next
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= d_ps->numParticles) return;

    // 1. Find which cell this particle belongs to
    const float inv_dx = 1.0f / d_fg->grid_spacing; // Assuming dx is the member name
    int i = floorf((d_ps->x[particle_idx] - d_fg->originX) * inv_dx);
    int j = floorf((d_ps->y[particle_idx] - d_fg->originY) * inv_dx);
    int k = floorf((d_ps->z[particle_idx] - d_fg->originZ) * inv_dx);

    // Safety check: only process particles inside the grid
    if (i >= 0 && i < d_fg->nx && j >= 0 && j < d_fg->ny && k >= 0 && k < d_fg->nz) {

        int cell_idx = cc_idx(i, j, k, d_fg->nx, d_fg->ny);

        // 2. Atomically insert this particle at the head of the list for this cell.
        // `atomicExch` writes particle_idx to d_cell_heads[cell_idx] and returns
        // the value that was previously there.
        int old_head = atomicExch(&d_cell_heads[cell_idx], particle_idx);

        // 3. The old head is now the next particle in the list after this one.
        d_particle_next[particle_idx] = old_head;
    } else {
        // If the particle is outside the grid, it doesn't belong to any list.
        d_particle_next[particle_idx] = -1;
    }
}


/**
 * @brief Marks grid cells as FLUID if they contain one or more particles.
 *        This runs after the spatial hash grid has been built.
 *
 * @param d_fg         Pointer to the fluid grid.
 * @param d_cell_heads The head-of-list array from the spatial hash.
 */
__global__ void mark_fluid_cells_kernel(FluidGrid* d_fg, const int* d_cell_heads) {
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int cell_idx = initial_idx; cell_idx < d_fg->size_cellCentered; cell_idx += stride) {
        // If the linked list for this cell is not empty, mark it as a fluid cell.
        if (d_cell_heads[cell_idx] != -1) {
            if (d_fg->cellType[cell_idx] != SOLID_CELL) {
                d_fg->cellType[cell_idx] = FLUID_CELL;
            }
        }
    }
}


template<StaggeredGridType GridT>
__global__ void p2g_gather_kernel(
        ParticleSystem* d_ps, FluidGrid* d_fg,
        const int* d_cell_heads, const int* d_particle_next
) {
    // --- 1. Determine which grid node this thread is responsible for ---
    float* target_vel_array;
    int total_nodes;

    if constexpr (GridT == U_VEL) {
        target_vel_array = d_fg->u;
        total_nodes = d_fg->size_u;
    } else if constexpr (GridT == V_VEL) {
        target_vel_array = d_fg->v;
        total_nodes = d_fg->size_v;
    } else { // W_VEL
        target_vel_array = d_fg->w;
        total_nodes = d_fg->size_w;
    }

    int initial_node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int node_idx = initial_node_idx; node_idx < total_nodes; node_idx += stride) {
        float velocity_sum = 0.0f;
        float weight_sum = 0.0f;

        const int nx = d_fg->nx;
        const int ny = d_fg->ny;
        const int nz = d_fg->nz;

        // --- 2. Unpack 1D node index to 3D grid coordinates (i, j, k) ---
        int i, j, k;
        if constexpr (GridT == U_VEL) {
            k = node_idx / ((nx + 1) * ny);
            int temp = node_idx % ((nx + 1) * ny);
            j = temp / (nx + 1);
            i = temp % (nx + 1);
        } else if constexpr (GridT == V_VEL) {
            k = node_idx / (nx * (ny + 1));
            int temp = node_idx % (nx * (ny + 1));
            j = temp / nx;
            i = temp % nx;
        } else { // W_VEL
            k = node_idx / (nx * ny);
            int temp = node_idx % (nx * ny);
            j = temp / nx;
            i = temp % nx;
        }

        // --- 3. Define the correct search neighborhood (THE FIX IS HERE) ---
        int i_min_offset, i_max_offset;
        int j_min_offset, j_max_offset;
        int k_min_offset, k_max_offset;

        if constexpr (GridT == U_VEL) { // 2x3x3 search
            i_min_offset = -1; i_max_offset = 0;
            j_min_offset = -1; j_max_offset = 1;
            k_min_offset = -1; k_max_offset = 1;
        } else if constexpr (GridT == V_VEL) { // 3x2x3 search
            i_min_offset = -1; i_max_offset = 1;
            j_min_offset = -1; j_max_offset = 0;
            k_min_offset = -1; k_max_offset = 1;
        } else { // W_VEL -- 3x3x2 search
            i_min_offset = -1; i_max_offset = 1;
            j_min_offset = -1; j_max_offset = 1;
            k_min_offset = -1; k_max_offset = 0;
        }

        // --- 4. Loop over the correct cell neighborhood ---
        for (int cell_k_offset = k_min_offset; cell_k_offset <= k_max_offset; ++cell_k_offset) {
            for (int cell_j_offset = j_min_offset; cell_j_offset <= j_max_offset; ++cell_j_offset) {
                for (int cell_i_offset = i_min_offset; cell_i_offset <= i_max_offset; ++cell_i_offset) {

                    int check_i = i + cell_i_offset;
                    int check_j = j + cell_j_offset;
                    int check_k = k + cell_k_offset;

                    if (check_i < 0 || check_i >= nx || check_j < 0 || check_j >= ny || check_k < 0 || check_k >= nz) continue;

                    int cell_idx = cc_idx(check_i, check_j, check_k, nx, ny);
                    int particle_idx = d_cell_heads[cell_idx];

                    // --- 5. Traverse linked list for this cell (The slow part) ---
                    while (particle_idx != -1) {
                        const float px = d_ps->x[particle_idx];
                        const float py = d_ps->y[particle_idx];
                        const float pz = d_ps->z[particle_idx];

                        const float inv_gs = 1.0f / d_fg->grid_spacing;
                        float particle_vel;
                        float p_gx, p_gy, p_gz; // Particle position in the specific staggered grid's coords

                        // Get the right velocity and calculate shifted coordinates
                        if constexpr (GridT == U_VEL) {
                            particle_vel = d_ps->vx[particle_idx];
                            p_gx = (px - d_fg->originX) * inv_gs;
                            p_gy = (py - d_fg->originY) * inv_gs - 0.5f;
                            p_gz = (pz - d_fg->originZ) * inv_gs - 0.5f;
                        } else if constexpr (GridT == V_VEL) {
                            particle_vel = d_ps->vy[particle_idx];
                            p_gx = (px - d_fg->originX) * inv_gs - 0.5f;
                            p_gy = (py - d_fg->originY) * inv_gs;
                            p_gz = (pz - d_fg->originZ) * inv_gs - 0.5f;
                        } else { // W_VEL
                            particle_vel = d_ps->vz[particle_idx];
                            p_gx = (px - d_fg->originX) * inv_gs - 0.5f;
                            p_gy = (py - d_fg->originY) * inv_gs - 0.5f;
                            p_gz = (pz - d_fg->originZ) * inv_gs;
                        }

                        // Trilinear weight calculation (this part was correct)
                        float wx = 1.0f - fabsf(p_gx - i);
                        float wy = 1.0f - fabsf(p_gy - j);
                        float wz = 1.0f - fabsf(p_gz - k);

                        if (wx > 0 && wy > 0 && wz > 0) {
                            float weight = wx * wy * wz;
                            velocity_sum += particle_vel * weight;
                            weight_sum += weight;
                        }
                        particle_idx = d_particle_next[particle_idx];
                    }
                }
            }
        }

        // --- 6. Finalize the velocity for this node ---
        if (weight_sum > 1e-6f) {
            target_vel_array[node_idx] = velocity_sum / weight_sum;
        } else {
            target_vel_array[node_idx] = 0.0f;
        }
    }
}


template __global__ void p2g_gather_kernel<U_VEL>(
        ParticleSystem* d_ps, FluidGrid* d_fg, const int* d_cell_heads, const int* d_particle_next);

template __global__ void p2g_gather_kernel<V_VEL>(
        ParticleSystem* d_ps, FluidGrid* d_fg, const int* d_cell_heads, const int* d_particle_next);

template __global__ void p2g_gather_kernel<W_VEL>(
        ParticleSystem* d_ps, FluidGrid* d_fg, const int* d_cell_heads, const int* d_particle_next);
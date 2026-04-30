/** @file cuda_kernels.cuh
 * @brief Header file for CUDA kernels used in the fluid simulator.
 *
 * This file contains declarations for CUDA kernels that perform various operations
 * on the fluid simulation data structures, such as initializing particles, updating
 * velocities, and enforcing boundary conditions. They are mostly listed in the order they
 * should be called in.
 */

#ifndef FLUID_SIMULATOR_CUDA_KERNELS_CUH
#define FLUID_SIMULATOR_CUDA_KERNELS_CUH

#include "data_structures.h"

// Define constants for cell types
const int EMPTY_CELL = 0;
const int FLUID_CELL = 1;
const int SOLID_CELL = 2;

// Define the grid type for the staggered grid
enum GridType {
    U_GRID,
    V_GRID,
    W_GRID
};


/**
 * @brief Helper function to get the 1D index for the U-velocity grid.
 * Dimensions: (nx+1, ny, nz)
 */
__device__ inline int u_idx(int i, int j, int k, int nx, int ny, int nz) {
    return i + j * (nx + 1) + k * (nx + 1) * ny;
}

/**
 * @brief Helper function to get the 1D index for the V-velocity grid.
 * Dimensions: (nx, ny+1, nz)
 */
__device__ inline int v_idx(int i, int j, int k, int nx, int ny, int nz) {
    return i + j * nx + k * nx * (ny + 1);
}

/**
 * @brief Helper function to get the 1D index for the W-velocity grid.
 * Dimensions: (nx, ny, nz+1)
 */
__device__ inline int w_idx(int i, int j, int k, int nx, int ny, int nz) {
    return i + j * nx + k * nx * ny;
}

/**
 * @brief Helper function to get the 1D index for the cell-centered grid.
 * Dimensions: (nx, ny, nz)
 */
__device__ inline int cc_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}


__global__ void initialize_particles_kernel(ParticleSystem* d_ps, float3 min_bounds, int3 particles_per_dim, float particle_spacing);

// __global__ void initialize_particles_kernel(ParticleSystem* d_ps);

// Most of the values are set when allocating, here we just fill the values like cell types.
__global__ void initialize_fluid_grid_kernel(FluidGrid* d_fg);

__global__ void update_particle_positions_kernel(ParticleSystem* d_ps, FluidGrid* d_fg, float dt);

__global__ void zero_out_fluid_grid_kernel(FluidGrid* d_fg);

__global__ void particle_to_grid_kernel(ParticleSystem* d_ps, FluidGrid* d_fg) ;

__global__ void normalize_grid_velocities_kernel(FluidGrid* d_fg);

__global__ void save_velocities_kernel(FluidGrid* d_fg);

__global__ void add_gravity_kernel(FluidGrid* d_fg, float gravity_y, float dt);

__global__ void enforce_boundary_kernel(FluidGrid* d_fg);

__global__ void calculate_divergence_kernel(FluidGrid* d_fg);

// This kernel needs fg_swap_pressure_pointers to be called after it.
__global__ void jacobi_iteration_kernel(FluidGrid* d_fg);

__global__ void project_velocities_kernel(FluidGrid* d_fg);

__global__ void grid_to_particle_kernel(ParticleSystem* d_ps, FluidGrid* d_fg, float alpha);

__global__ void update_pbo_kernel(ParticleSystem* d_ps, float* pbo_mapped_ptr);

// UNUSED: Kernels from here are experimental and not used in the current implementation
__global__ void build_spatial_hash_kernel( ParticleSystem* d_ps, FluidGrid* d_fg, int* d_cell_heads, int* d_particle_next );

__global__ void particle_to_grid_from_hash_kernel(
        ParticleSystem* d_ps, FluidGrid* d_fg,
        int* d_cell_heads, int* d_particle_next
);

__global__ void mark_fluid_cells_kernel(FluidGrid* d_fg, const int* d_cell_heads);

// Enum to identify grid type for the template
enum StaggeredGridType { U_VEL, V_VEL, W_VEL };

template<StaggeredGridType GridT>
__global__ void p2g_gather_kernel(ParticleSystem* d_ps, FluidGrid* d_fg, const int* d_cell_heads, const int* d_particle_next );

#endif // FLUID_SIMULATOR_CUDA_KERNELS_CUH

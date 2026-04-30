#include "cuda_kernels.cuh"

// Sort particles by grid cell before processing
__global__ void compute_particle_cell_keys(ParticleSystem* d_ps, FluidGrid* d_fg, int* keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_ps->numParticles) return;

    float gx = (d_ps->x[idx] - d_fg->originX) / d_fg->grid_spacing;
    float gy = (d_ps->y[idx] - d_fg->originY) / d_fg->grid_spacing;
    float gz = (d_ps->z[idx] - d_fg->originZ) / d_fg->grid_spacing;

    int cell_i = floorf(gx);
    int cell_j = floorf(gy);
    int cell_k = floorf(gz);

    keys[idx] = cell_i + cell_j * d_fg->nx + cell_k * d_fg->nx * d_fg->ny;
}

// Then use thrust::sort_by_key to sort particles by cell


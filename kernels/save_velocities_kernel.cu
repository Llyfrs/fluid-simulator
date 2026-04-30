#include "cuda_kernels.cuh"

/**
 * @brief Saves the current grid velocities into the temporary buffer arrays.
 *        This is done before applying forces and projection.
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */
__global__ void save_velocities_kernel(FluidGrid* d_fg) {
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = initial_idx; i < d_fg->size_u; i += stride) {
        d_fg->u_temp[i] = d_fg->u[i];
    }
    for (int i = initial_idx; i < d_fg->size_v; i += stride) {
        d_fg->v_temp[i] = d_fg->v[i];
    }
    for (int i = initial_idx; i < d_fg->size_w; i += stride) {
        d_fg->w_temp[i] = d_fg->w[i];
    }
}
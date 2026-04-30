#include "cuda_kernels.cuh"



/**
 * @brief Zeros out the velocity and weight arrays of the fluid grid using a
 *        grid-stride loop. This allows a modest number of threads to clear
 *        very large arrays efficiently.
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */
__global__ void zero_out_fluid_grid_kernel(FluidGrid* d_fg) {
    // Calculate the unique global index for this thread
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the total number of threads in the entire grid launch
    int stride = gridDim.x * blockDim.x;

    // --- Loop 1: Zero out the U and U-weights arrays ---
    for (int i = initial_idx; i < d_fg->size_u; i += stride) {
        d_fg->u[i] = 0.0f;
        d_fg->u_weights[i] = 0.0f;
    }

    // --- Loop 2: Zero out the V and V-weights arrays ---
    for (int i = initial_idx; i < d_fg->size_v; i += stride) {
        d_fg->v[i] = 0.0f;
        d_fg->v_weights[i] = 0.0f;
    }

    // --- Loop 3: Zero out the W and W-weights arrays ---
    for (int i = initial_idx; i < d_fg->size_w; i += stride) {
        d_fg->w[i] = 0.0f;
        d_fg->w_weights[i] = 0.0f;
    }

    for (int i = initial_idx; i < d_fg->size_cellCentered; i += stride) {
        // If the cell is not a permanent solid wall, reset it to empty.
        if (d_fg->cellType[i] != SOLID_CELL) {
            d_fg->cellType[i] = EMPTY_CELL;
        }
    }
}
#include "cuda_kernels.cuh"

/**
 * @brief Initializes the entire fluid grid on the device. This kernel should be
 *        run only ONCE after the grid is allocated. It zeros out all data arrays
 *        and sets up the solid boundaries.
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */
__global__ void initialize_fluid_grid_kernel(FluidGrid* d_fg) {
    // Standard grid-stride setup
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // --- Initialize U-sized arrays ---
    for (int i = initial_idx; i < d_fg->size_u; i += stride) {
        d_fg->u[i] = 0.0f;
        d_fg->u_weights[i] = 0.0f;
        d_fg->u_temp[i] = 0.0f; // Also zero out the temporary buffer
    }

    // --- Initialize V-sized arrays ---
    for (int i = initial_idx; i < d_fg->size_v; i += stride) {
        d_fg->v[i] = 0.0f;
        d_fg->v_weights[i] = 0.0f;
        d_fg->v_temp[i] = 0.0f;
    }

    // --- Initialize W-sized arrays ---
    for (int i = initial_idx; i < d_fg->size_w; i += stride) {
        d_fg->w[i] = 0.0f;
        d_fg->w_weights[i] = 0.0f;
        d_fg->w_temp[i] = 0.0f;
    }

    // --- Initialize all cell-centered arrays and set boundaries ---
    for (int idx = initial_idx; idx < d_fg->size_cellCentered; idx += stride) {
        // 1. Zero out all relevant float arrays
        d_fg->pressure[idx] = 0.0f;
        d_fg->divergence[idx] = 0.0f;
        d_fg->phi[idx] = 0.0f;

        // 2. Initialize the entire domain as empty/air first

        d_fg->cellType[idx] = EMPTY_CELL;

        // 3. Identify and set the solid boundary cells
        // Unpack the 1D index back to 3D (i, j, k) coordinates
        int k = idx / (d_fg->nx * d_fg->ny);
        int temp = idx % (d_fg->nx * d_fg->ny);
        int j = temp / d_fg->nx;
        int i = temp % d_fg->nx;

        // Check if the cell is on the outermost border of the domain
        if (i == 0 || i == d_fg->nx - 1 ||
            j == 0 || j == d_fg->ny - 1 ||
            k == 0 || k == d_fg->nz - 1) {
            // Overwrite the type to SOLID for boundary cells
            d_fg->cellType[idx] = SOLID_CELL;
        }
    }
}
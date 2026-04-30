#include "cuda_kernels.cuh"

/**
 * @brief Calculates the divergence of the velocity field for each fluid cell.
 *        Divergence = dU/dx + dV/dy + dW/dz
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */

__global__ void calculate_divergence_kernel(FluidGrid* d_fg) {
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const int nx = d_fg->nx;
    const int ny = d_fg->ny;
    const int nz = d_fg->nz;
    const float inv_dx = 1.0f / d_fg->grid_spacing;

    for (int idx = initial_idx; idx < d_fg->size_cellCentered; idx += stride) {
        // Only compute divergence for fluid cells
        if (d_fg->cellType[idx] == FLUID_CELL) {
            // Unpack 1D index to 3D (i, j, k)
            int k = idx / (nx * ny);
            int temp = idx % (nx * ny);
            int j = temp / nx;
            int i = temp % nx;

            // Get the velocities on the faces of this cell
            float u_right = d_fg->u[u_idx(i + 1, j, k, nx, ny, nz)];
            float u_left  = d_fg->u[u_idx(i,     j, k, nx, ny, nz)];
            float v_top   = d_fg->v[v_idx(i, j + 1, k, nx, ny, nz)];
            float v_bot   = d_fg->v[v_idx(i,     j, k, nx, ny, nz)];
            float w_front = d_fg->w[w_idx(i, j, k + 1, nx, ny, nz)];
            float w_back  = d_fg->w[w_idx(i, j,     k, nx, ny, nz)];

            // Central difference formula for divergence
            float div = (u_right - u_left) + (v_top - v_bot) + (w_front - w_back);
            d_fg->divergence[idx] = div * inv_dx;
        } else {
            d_fg->divergence[idx] = 0.0f;
        }
    }
}
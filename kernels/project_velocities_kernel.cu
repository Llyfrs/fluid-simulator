#include "cuda_kernels.cuh"

/**
 * @brief Subtracts the pressure gradient from the velocity field to make it
 *        divergence-free (the "projection" step).
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */
__global__ void project_velocities_kernel(FluidGrid* d_fg) {
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const int nx = d_fg->nx;
    const int ny = d_fg->ny;
    const int nz = d_fg->nz;
    const float inv_dx = 1.0f / d_fg->grid_spacing;

    // --- Project U-velocities ---
    for (int i = initial_idx; i < d_fg->size_u; i += stride) {
        int k = i / ((nx + 1) * ny);
        int temp = i % ((nx + 1) * ny);
        int j = temp / (nx + 1);
        int grid_i = temp % (nx + 1);

        if (grid_i > 0 && grid_i < nx) { // Only operate on internal faces
            int p_idx_right = cc_idx(grid_i, j, k, nx, ny);
            int p_idx_left  = cc_idx(grid_i - 1, j, k, nx, ny);
            // Only subtract if one of the cells is a fluid cell
            if (d_fg->cellType[p_idx_left] == FLUID_CELL || d_fg->cellType[p_idx_right] == FLUID_CELL) {
                d_fg->u[i] -= (d_fg->pressure[p_idx_right] - d_fg->pressure[p_idx_left]) * inv_dx;
            }
        }
    }

    // --- Project V-velocities ---
    for (int i = initial_idx; i < d_fg->size_v; i += stride) {
        int k = i / (nx * (ny + 1));
        int temp = i % (nx * (ny + 1));
        int j = temp / nx;
        int grid_i = temp % nx;

        if (j > 0 && j < ny) { // Only operate on internal faces
            int p_idx_top = cc_idx(grid_i, j, k, nx, ny);
            int p_idx_bot = cc_idx(grid_i, j - 1, k, nx, ny);
            // Only subtract if one of the cells is a fluid cell
            if (d_fg->cellType[p_idx_top] == FLUID_CELL || d_fg->cellType[p_idx_bot] == FLUID_CELL) {
                d_fg->v[i] -= (d_fg->pressure[p_idx_top] - d_fg->pressure[p_idx_bot]) * inv_dx;
            }
        }
    }

    // --- Project W-velocities ---
    for (int i = initial_idx; i < d_fg->size_w; i += stride) {
        int k = i / (nx * ny);
        int temp = i % (nx * ny);
        int j = temp / nx;
        int grid_i = temp % nx;

        if (k > 0 && k < nz) { // Only operate on internal faces
            int p_idx_front = cc_idx(grid_i, j, k, nx, ny);
            int p_idx_back  = cc_idx(grid_i, j, k - 1, nx, ny);
            // Only subtract if one of the cells is a fluid cell
            if (d_fg->cellType[p_idx_front] == FLUID_CELL || d_fg->cellType[p_idx_back] == FLUID_CELL) {
                d_fg->w[i] -= (d_fg->pressure[p_idx_front] - d_fg->pressure[p_idx_back]) * inv_dx;
            }
        }
    }
}
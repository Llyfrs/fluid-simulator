#include "cuda_kernels.cuh"

/**
 * @brief Performs one PURE Jacobi iteration. It reads pressure values from
 *        d_fg->pressure and writes the newly calculated values into d_fg->pressure_temp.
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */
__global__ void jacobi_iteration_kernel(FluidGrid* d_fg) {

    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const int nx = d_fg->nx;
    const int ny = d_fg->ny;
    const int nz = d_fg->nz;
    const float dx_sq = d_fg->grid_spacing * d_fg->grid_spacing; // Assuming 'dx' is the member name for grid spacing

    for (int idx = initial_idx; idx < d_fg->size_cellCentered; idx += stride) {
        // We only perform the calculation for fluid cells.
        if (d_fg->cellType[idx] == FLUID_CELL) {
            int k = idx / (nx * ny);
            int temp = idx % (nx * ny);
            int j = temp / nx;
            int i = temp % nx;

            float sum_neighbors_pressure = 0.0f;

            // Left Neighbor (i-1)
            if (i > 0 && d_fg->cellType[cc_idx(i - 1, j, k, nx, ny)] != SOLID_CELL)
                sum_neighbors_pressure += d_fg->pressure[cc_idx(i - 1, j, k, nx, ny)];
            else if (i > 0) // It's a solid wall
                sum_neighbors_pressure += d_fg->pressure[idx]; // Use THIS cell's current pressure

            // Right Neighbor (i+1)
            if (i < nx - 1 && d_fg->cellType[cc_idx(i + 1, j, k, nx, ny)] != SOLID_CELL)
                sum_neighbors_pressure += d_fg->pressure[cc_idx(i + 1, j, k, nx, ny)];
            else if (i < nx - 1)
                sum_neighbors_pressure += d_fg->pressure[idx];

            // Bottom Neighbor (j-1)
            if (j > 0 && d_fg->cellType[cc_idx(i, j - 1, k, nx, ny)] != SOLID_CELL)
                sum_neighbors_pressure += d_fg->pressure[cc_idx(i, j - 1, k, nx, ny)];
            else if (j > 0)
                sum_neighbors_pressure += d_fg->pressure[idx];

            // Top Neighbor (j+1)
            if (j < ny - 1 && d_fg->cellType[cc_idx(i, j + 1, k, nx, ny)] != SOLID_CELL)
                sum_neighbors_pressure += d_fg->pressure[cc_idx(i, j + 1, k, nx, ny)];
            else if (j < ny - 1)
                sum_neighbors_pressure += d_fg->pressure[idx];

            // Back Neighbor (k-1)
            if (k > 0 && d_fg->cellType[cc_idx(i, j, k - 1, nx, ny)] != SOLID_CELL)
                sum_neighbors_pressure += d_fg->pressure[cc_idx(i, j, k - 1, nx, ny)];
            else if (k > 0)
                sum_neighbors_pressure += d_fg->pressure[idx];

            // Front Neighbor (k+1)
            if (k < nz - 1 && d_fg->cellType[cc_idx(i, j, k + 1, nx, ny)] != SOLID_CELL)
                sum_neighbors_pressure += d_fg->pressure[cc_idx(i, j, k + 1, nx, ny)];
            else if (k < nz - 1)
                sum_neighbors_pressure += d_fg->pressure[idx];

            // Calculate the new pressure based on the sum of OLD neighbors
            float new_pressure = (sum_neighbors_pressure - d_fg->divergence[idx] * dx_sq) / 6.0f;

            // --- THE WRITE goes to d_fg->pressure_temp ---
            d_fg->temp_pressure[idx] = new_pressure;

        } else {
            // If it's not a fluid cell, its pressure is 0. Write this to the temp buffer.
            d_fg->temp_pressure[idx] = 0.0f;
        }
    }
}
#include "cuda_kernels.cuh"


/**
 * @brief Normalizes the grid velocities by dividing the summed velocities by the
 *        summed weights using a grid-stride loop. This turns the sums into
 *        averages and handles grid nodes with no particle influence.
 *
 * @param d_fg Pointer to the fluid grid struct on the device.
 */
__global__ void normalize_grid_velocities_kernel(FluidGrid* d_fg) {
    // A small value to prevent division by zero.
    const float epsilon = 1e-9f;

    // Calculate the unique global index for this thread.
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the total number of threads in the entire grid launch.
    int stride = gridDim.x * blockDim.x;

    // --- Loop 1: Normalize U velocities ---
    for (int i = initial_idx; i < d_fg->size_u; i += stride) {
        // If the total weight is significant, perform the division.
        if (d_fg->u_weights[i] > epsilon) {
            d_fg->u[i] /= d_fg->u_weights[i];
        }
        // If the weight is zero, set the velocity to zero to avoid NaN.
        else {
            d_fg->u[i] = 0.0f;
        }
    }

    // --- Loop 2: Normalize V velocities ---
    for (int i = initial_idx; i < d_fg->size_v; i += stride) {
        if (d_fg->v_weights[i] > epsilon) {
            d_fg->v[i] /= d_fg->v_weights[i];
        }
        // If the weight is zero, set the velocity to zero to avoid NaN.
        else {
            d_fg->v[i] = 0.0f;
        }
    }

    // --- Loop 3: Normalize W velocities ---
    for (int i = initial_idx; i < d_fg->size_w; i += stride) {
        if (d_fg->w_weights[i] > epsilon) {
            d_fg->w[i] /= d_fg->w_weights[i];
        }
        // If the weight is zero, set the velocity to zero to avoid NaN.
        else {
            d_fg->w[i] = 0.0f;
        }
    }
}
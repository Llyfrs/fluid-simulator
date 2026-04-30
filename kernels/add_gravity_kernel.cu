#include "cuda_kernels.cuh"

/**
 * @brief Applies a constant gravitational force to the V-velocity component of the grid.
 *        Uses a grid-stride loop to process the entire v-array.
 *
 * @param d_fg      Pointer to the fluid grid struct on the device.
 * @param gravity_y The acceleration due to gravity (e.g., -9.8f).
 */
__global__ void add_gravity_kernel(FluidGrid* d_fg, float gravity_y, float dt) {
    // Calculate the unique global index for this thread.
    int initial_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the total number of threads in the entire grid launch.
    int stride = gridDim.x * blockDim.x;

    // The change in velocity for this timestep.
    float delta_v = gravity_y;

    // Loop over the V-velocity array using a grid-stride.
    for (int i = initial_idx; i < d_fg->size_v; i += stride) {
        d_fg->v[i] += delta_v;
    }
}
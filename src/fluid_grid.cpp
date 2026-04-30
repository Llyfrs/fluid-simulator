#include "data_structures.h"
#include "cuda_helpers.h"
#include "fluid_grid.h"
#include <cstdio>
#include <cstdlib>

FluidGrid* fg_allocate_device(int nx, int ny, int nz, float dx, float originX, float originY, float originZ) {
    // 1. Allocate memory for the FluidGrid struct itself on the device
    FluidGrid* d_grid_ptr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_grid_ptr, sizeof(FluidGrid)));

    // 2. Create a temporary host-side FluidGrid struct.
    //    This struct will be filled with DEVICE pointers and scalar values,
    //    and then copied as a single block to the device.
    FluidGrid temp_host_grid_with_device_pointers;

    // 3. Calculate array sizes based on the staggered MAC grid layout
    size_t size_u = (size_t)(nx + 1) * ny * nz;
    size_t size_v = (size_t)nx * (ny + 1) * nz;
    size_t size_w = (size_t)nx * ny * (nz + 1);
    size_t size_cellCentered = (size_t)nx * ny * nz;

    // Calculate byte sizes for allocations
    size_t u_bytes = size_u * sizeof(float);
    size_t v_bytes = size_v * sizeof(float);
    size_t w_bytes = size_w * sizeof(float);
    size_t cc_float_bytes = size_cellCentered * sizeof(float); // Cell-centered float arrays
    size_t cc_int_bytes = size_cellCentered * sizeof(int);     // Cell-centered int arrays

    // 4. Allocate memory for all device arrays and store their pointers
    //    in our temporary host struct.
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.u, u_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.v, v_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.w, w_bytes));

    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.r , cc_float_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.d , cc_float_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.q , cc_float_bytes));

    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.u_weights, u_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.v_weights, v_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.w_weights, w_bytes));

    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.u_temp, u_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.v_temp, v_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.w_temp, w_bytes));

    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.pressure, cc_float_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.divergence, cc_float_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.phi, cc_float_bytes));

    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.temp_pressure, cc_float_bytes));

    checkCudaErrors(cudaMalloc((void**)&temp_host_grid_with_device_pointers.cellType, cc_int_bytes));

    // 5. Set the scalar properties in the temporary host struct
    temp_host_grid_with_device_pointers.nx = nx;
    temp_host_grid_with_device_pointers.ny = ny;
    temp_host_grid_with_device_pointers.nz = nz;
    temp_host_grid_with_device_pointers.grid_spacing = dx;
    temp_host_grid_with_device_pointers.originX = originX;
    temp_host_grid_with_device_pointers.originY = originY;
    temp_host_grid_with_device_pointers.originZ = originZ;
    temp_host_grid_with_device_pointers.size_u = size_u;
    temp_host_grid_with_device_pointers.size_v = size_v;
    temp_host_grid_with_device_pointers.size_w = size_w;
    temp_host_grid_with_device_pointers.size_cellCentered = size_cellCentered;

    // 6. Copy the temporary host struct (which now contains device pointers
    //    and scalar values) to the allocated device FluidGrid struct.
    checkCudaErrors(cudaMemcpy(d_grid_ptr, &temp_host_grid_with_device_pointers, sizeof(FluidGrid), cudaMemcpyHostToDevice));

    // Prints the total size of the FluidGrid allocated on the device
    size_t total_size = sizeof(FluidGrid) +
                        u_bytes + v_bytes + w_bytes +
                        3 * u_bytes + 3 * v_bytes + 3 * w_bytes + // Weights and temp arrays
                        3 * cc_float_bytes + cc_int_bytes; // Pressure, divergence, phi, cellType
    printf("Allocated FluidGrid on device with total size: %zu MB\n", total_size / (1024 * 1024));

    // Return the device pointer to the fully configured FluidGrid struct
    return d_grid_ptr;
}


void fg_swap_pressure_pointers(FluidGrid* d_grid) {

    if (d_grid == nullptr) {
        return;
    }

    // 1. Create a temporary host-side struct to hold the device pointers.
    FluidGrid temp_host_grid;

    // 2. Copy the device-side struct to the host to get the pointers.
    checkCudaErrors(cudaMemcpy(&temp_host_grid, d_grid, sizeof(FluidGrid), cudaMemcpyDeviceToHost));

    // 3. Swap the pressure and temp_pressure pointers
    float* temp = temp_host_grid.pressure;
    temp_host_grid.pressure = temp_host_grid.temp_pressure;
    temp_host_grid.temp_pressure = temp;

    // 4. Copy the modified host struct back to the device
    checkCudaErrors(cudaMemcpy(d_grid, &temp_host_grid, sizeof(FluidGrid), cudaMemcpyHostToDevice));
}

void fg_save_velocity(FluidGrid* d_grid) {
    if (d_grid == nullptr) {
        return;
    }

    // 1. Create a temporary host-side struct to hold the device pointers.
    FluidGrid temp_host_grid;

    // 2. Copy the device-side struct to the host to get the pointers.
    checkCudaErrors(cudaMemcpy(&temp_host_grid, d_grid, sizeof(FluidGrid), cudaMemcpyDeviceToHost));

    // 3. Copy memory from device to device for u, v, w to their temporary arrays
    checkCudaErrors(cudaMemcpy(temp_host_grid.u_temp, temp_host_grid.u, temp_host_grid.size_u * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(temp_host_grid.v_temp, temp_host_grid.v, temp_host_grid.size_v * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(temp_host_grid.w_temp, temp_host_grid.w, temp_host_grid.size_w * sizeof(float), cudaMemcpyDeviceToDevice));

    // 4. Copy the modified host struct back to the device
    checkCudaErrors(cudaMemcpy(d_grid, &temp_host_grid, sizeof(FluidGrid), cudaMemcpyHostToDevice));


}

void fg_freeMem_device(FluidGrid* d_grid) {
    if (d_grid == nullptr) {
        return;
    }

    // 1. Create a temporary host-side struct to hold the device pointers.
    FluidGrid temp_host_grid;

    // 2. Copy the device-side struct to the host to get the pointers.
    checkCudaErrors(cudaMemcpy(&temp_host_grid, d_grid, sizeof(FluidGrid), cudaMemcpyDeviceToHost));

    // 3. Free each of the internal arrays on the device.
    checkCudaErrors(cudaFree(temp_host_grid.u));
    checkCudaErrors(cudaFree(temp_host_grid.v));
    checkCudaErrors(cudaFree(temp_host_grid.w));
    checkCudaErrors(cudaFree(temp_host_grid.u_temp));
    checkCudaErrors(cudaFree(temp_host_grid.v_temp));
    checkCudaErrors(cudaFree(temp_host_grid.w_temp));
    checkCudaErrors(cudaFree(temp_host_grid.pressure));
    checkCudaErrors(cudaFree(temp_host_grid.divergence));
    checkCudaErrors(cudaFree(temp_host_grid.phi));
    checkCudaErrors(cudaFree(temp_host_grid.cellType));

    checkCudaErrors(cudaFree(temp_host_grid.u_weights));
    checkCudaErrors(cudaFree(temp_host_grid.v_weights));
    checkCudaErrors(cudaFree(temp_host_grid.w_weights));
    checkCudaErrors(cudaFree(temp_host_grid.temp_pressure));
    checkCudaErrors(cudaFree(temp_host_grid.r));
    checkCudaErrors(cudaFree(temp_host_grid.d));
    checkCudaErrors(cudaFree(temp_host_grid.q));



    // 4. Finally, free the device struct itself.
    checkCudaErrors(cudaFree(d_grid));
}

#include "particle_system.h"
#include <cstdio>
#include <cstdlib>
#include "cuda_helpers.h"

// Sanity check for null pointers
#define ASSERT_ALLOC(ptr) if (!(ptr)) { fprintf(stderr, "Null pointer: %s\n", #ptr); exit(EXIT_FAILURE); }



#define M_PI 3.14159265358979323846

ParticleSystem* ps_allocate_device(int num_particles) {

    // 1. Allocate memory for the ParticleSystem struct itself on the device
    ParticleSystem* d_ps_ptr = nullptr; // This will be the pointer to the ParticleSystem struct on the device
    checkCudaErrors(cudaMalloc((void**)&d_ps_ptr, sizeof(ParticleSystem)));

    // 2. Create a temporary host-side ParticleSystem struct.
    //    This struct will be filled with DEVICE pointers from cudaMalloc
    //    and then copied entirely to the d_ps_ptr on the device.
    ParticleSystem temp_host_ps_with_device_pointers;

    // 3. Allocate memory for the device particle arrays
    //    Store these device pointers in our temporary host struct
    size_t array_size_bytes = num_particles * sizeof(float);

    // Print memory allocation size for debugging
    size_t total_size_bytes = sizeof(ParticleSystem) +
                              6 * array_size_bytes; // 6 arrays: x, y, z, vx, vy, vz

    printf("Trying to allocated ParticleSystem on device with total size: %zu MB\n", total_size_bytes / (1024 * 1024));


    checkCudaErrors(cudaMalloc((void**)&temp_host_ps_with_device_pointers.x, array_size_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_ps_with_device_pointers.y, array_size_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_ps_with_device_pointers.z, array_size_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_ps_with_device_pointers.vx, array_size_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_ps_with_device_pointers.vy, array_size_bytes));
    checkCudaErrors(cudaMalloc((void**)&temp_host_ps_with_device_pointers.vz, array_size_bytes));

    // 4. Set the particle counts in the temporary host struct
    temp_host_ps_with_device_pointers.numParticles = num_particles;

    // 5. Copy the temporary host ParticleSystem (which now contains
    //    device pointers and counts) to the allocated device ParticleSystem struct.
    checkCudaErrors(cudaMemcpy(d_ps_ptr, &temp_host_ps_with_device_pointers, sizeof(ParticleSystem), cudaMemcpyHostToDevice));

    printf("Allocation of ParticleSystem on device successful.\n");

    return d_ps_ptr; // Return the device pointer to the ParticleSystem struct
}


void ps_freeMem_host(ParticleSystem& ps) {
    // Free allocated memory
    if (ps.x) free(ps.x); ps.x = nullptr;
    if (ps.y) free(ps.y); ps.y = nullptr;
    if (ps.z) free(ps.z); ps.z = nullptr;
    if (ps.vx) free(ps.vx); ps.vx = nullptr;
    if (ps.vy) free(ps.vy); ps.vy = nullptr;
    if (ps.vz) free(ps.vz); ps.vz = nullptr;

    // Reset particle count
    ps.numParticles = 0;

}

void ps_freeMem_device(ParticleSystem* d_ps) {

    if (d_ps == nullptr) {
        fprintf(stderr, "ERROR: Attempted to free null ParticleSystem device pointer.\n");
        return;
    }

    // 1. Create a temporary host-side ParticleSystem struct to hold device pointers
    ParticleSystem temp_host_ps;

    // 2. Copy the device-side struct to the host to get the pointers
    checkCudaErrors(cudaMemcpy(&temp_host_ps, d_ps, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));

    // 3. Free each device array
    checkCudaErrors(cudaFree(temp_host_ps.x));
    checkCudaErrors(cudaFree(temp_host_ps.y));
    checkCudaErrors(cudaFree(temp_host_ps.z));
    checkCudaErrors(cudaFree(temp_host_ps.vx));
    checkCudaErrors(cudaFree(temp_host_ps.vy));
    checkCudaErrors(cudaFree(temp_host_ps.vz));

    // 4. Finally, free the ParticleSystem struct itself
    checkCudaErrors(cudaFree(d_ps));

    printf("Freed ParticleSystem device memory.\n");
}

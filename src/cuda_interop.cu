#include "cuda_interop.h"
#include "globals.h"    // For g_cudaData, g_glData, gh_particleSystem
#include "config.h"     // For SIMULATION_TIME_STEP
#include "particle_system.h" // For ps_updatePosBuffer and ps_cpu_simulation_step (if used as placeholder)
#include <cuda_gl_interop.h>
#include "cuda_kernels.cuh"
#include "cuda_helpers.h"
#include <cstdio>      // For printf, fprintf
#include "fluid_grid.h"

void cuda_init_runtime() {
    printf("Initializing CUDA...\n");



    // --- 1. Initialize CUDA Device ---

    cudaSetDevice(0);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        exit(0);
    }
    printf("Found %d CUDA devices.\n", deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using device: %s\n", deviceProp.name);


    printf("Calculated particle requirements: %d total particles.\n", NUM_PARTICLES);

    // --- 3. Allocate Device Memory ---
    gd_particleSystem = ps_allocate_device(NUM_PARTICLES);
    if (!gd_particleSystem) {
        fprintf(stderr, "Failed to allocate device particle system for %d particles\n", NUM_PARTICLES);
        return;
    }
    printf("Device particle system allocated\n");

    gd_fluidGrid = fg_allocate_device(GRID_NX, GRID_NY, GRID_NZ, GRID_SPACING, GRID_ORIGIN.x, GRID_ORIGIN.y, GRID_ORIGIN.z);
    if (!gd_fluidGrid) {
        fprintf(stderr, "Failed to allocate device fluid grid\n");
        ps_freeMem_device(gd_particleSystem); // Clean up already allocated memory
        return;
    }
    printf("Fluid grid allocated\n");


    // --- 4. Initialize Data on the Device using Kernels ---
    // Initialize Particles
    int particle_threads = 256;
    int particle_blocks = (NUM_PARTICLES + particle_threads - 1) / particle_threads;
    printf("Initializing particles with %d blocks, %d threads\n", particle_blocks, particle_threads);

    // Use the NEW kernel with all its parameters
    initialize_particles_kernel<<<particle_blocks, particle_threads>>>(
            gd_particleSystem,
            FLUID_MIN_BOUNDS,
            PARTICLES_PER_DIMENSION,
            PARTICLE_SPACING
    );

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    printf("Particles initialized\n");

    // Initialize Fluid Grid (using a grid-stride friendly launch configuration)
    int grid_init_threads = 256;
    int grid_init_blocks = 128; // This is a good general purpose size

    initialize_fluid_grid_kernel<<<grid_init_blocks, grid_init_threads>>>(gd_fluidGrid);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    printf("Fluid grid initialized with solid boundaries\n");


    // --- 5. Register with OpenGL (no changes needed here) ---
    if (g_glData.pboID > 0) {
        cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&g_cudaData.pboResource, g_glData.pboID, cudaGraphicsRegisterFlagsWriteDiscard);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed for PBO: %s\n", cudaGetErrorString(cudaStatus));
            g_cudaData.pboResource = nullptr;
        } else {
            printf("Registered OpenGL VBO with CUDA (PBO ID: %u)\n", g_glData.pboID);
        }
    } else {
        fprintf(stderr, "Warning: PBO ID is 0, CUDA registration skipped.\n");
        g_cudaData.pboResource = nullptr;
    }

    printf("CUDA Initialized.\n");
}



void cuda_run_simulation_step() {
    static int step_count = 0;
    static float total_time_ms = 0.0f;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    // 1. Map the PBO resource for CUDA to access
    float* d_pboMappedPtr = nullptr;
    size_t mappedSize;
    cudaError_t status = cudaGraphicsMapResources(1, &g_cudaData.pboResource, 0);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(status));
        return;
    }

    int threads = NUM_THREADS;
    int blocks = (NUM_PARTICLES + threads - 1) / threads;

    float simulation_dt = SIMULATION_TIME_STEP;

    for (int j = 0; j < SIMULATIONS_PER_FRAME; ++j) {
        timeKernelExecution(
                { launchKernel((update_particle_positions_kernel<<<blocks, threads>>>(gd_particleSystem, gd_fluidGrid, simulation_dt))); },
                "update_particle_positions"
        );

        timeKernelExecution(
                { launchKernel((zero_out_fluid_grid_kernel<<<blocks, threads>>>(gd_fluidGrid))); },
                "zero_out_fluid_grid"
        );

        timeKernelExecution(
                { launchKernel((particle_to_grid_kernel<<<blocks, threads>>>(gd_particleSystem, gd_fluidGrid))); },
                "particle_to_grid"
        );

        timeKernelExecution(
                { launchKernel((normalize_grid_velocities_kernel<<<blocks, threads>>>(gd_fluidGrid))); },
                "normalize_grid_velocities"
        );

        timeKernelExecution(
                { launchKernel((save_velocities_kernel<<<blocks, threads>>>(gd_fluidGrid))); },
                "save_velocities"
        );

        timeKernelExecution(
                { launchKernel((add_gravity_kernel<<<blocks, threads>>>(gd_fluidGrid, GRAVITY, simulation_dt))); },
                "add_gravity"
        );

        timeKernelExecution(
                { launchKernel((enforce_boundary_kernel<<<blocks, threads>>>(gd_fluidGrid))); },
                "enforce_boundary"
        );

        timeKernelExecution(
                { launchKernel((calculate_divergence_kernel<<<blocks, threads>>>(gd_fluidGrid))); },
                "calculate_divergence"
        );

        timeKernelExecution(
                {
                    for (int i = 0; i < NUM_JACOBI_ITERATIONS; ++i) {
                        launchKernel((jacobi_iteration_kernel<<<blocks, threads>>>(gd_fluidGrid)));
                        fg_swap_pressure_pointers(gd_fluidGrid);
                    }
                },
                "jacobi_iteration_loop"
        );

        timeKernelExecution(
                { launchKernel((project_velocities_kernel<<<blocks, threads>>>(gd_fluidGrid))); },
                "project_velocities"
        );

        timeKernelExecution(
                { launchKernel((grid_to_particle_kernel<<<blocks, threads>>>(gd_particleSystem, gd_fluidGrid, BLEND))); },
                "grid_to_particle"
        );
    }

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pboMappedPtr, &mappedSize, g_cudaData.pboResource));

    timeKernelExecution(
            { launchKernel((update_pbo_kernel<<<blocks, threads>>>(gd_particleSystem, d_pboMappedPtr))); },
            "update_pbo"
    );

    // 3. Unmap the PBO resource
    checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cudaData.pboResource, 0));

    // 4. Time measurement
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTimeMs = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMs, start, stop));
    float deltaTime = elapsedTimeMs / 10000.0f;  // in seconds

    // Clean up
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // 5. FPS and deltaTime reporting
    ++step_count;
    total_time_ms += elapsedTimeMs;

    if (step_count % 60 == 0) {
        float avg_time_per_frame = total_time_ms / step_count;
        float fps = 1000.0f / avg_time_per_frame;
        printf("Average FPS over %d frames: %.2f (%.2f ms/frame) | Last Δt: %.4f sec\n",
               step_count, fps, avg_time_per_frame, deltaTime);
        step_count = 0;
        total_time_ms = 0.0f;
    }
}


void cuda_release_runtime() {
    printf("Releasing CUDA resources...\n");

    if (g_cudaData.pboResource) {
        cudaError_t cudaStatus = cudaGraphicsUnregisterResource(g_cudaData.pboResource);
        if (cudaStatus != cudaSuccess && cudaStatus != cudaErrorInvalidValue && cudaStatus != cudaErrorUnknown) {
            fprintf(stderr, "cudaGraphicsUnregisterResource failed: %s\n", cudaGetErrorString(cudaStatus));
        } else {
            printf(" - Unregistered PBO resource.\n");
        }
        g_cudaData.pboResource = nullptr;
    }

    ps_freeMem_device(gd_particleSystem); // Free device memory for particle system
    fg_freeMem_device(gd_fluidGrid);

    cudaDeviceReset();

    printf("CUDA resources released.\n");
}



/** * @file cuda_helpers.h
 * @brief Contains CUDA helper macros that make running kernels and checking errors easier.
 *
 * This file defines macros for error checking and kernel launching in CUDA.
 * It includes error handling for CUDA API calls, kernel launches, and timing kernel execution.
 */

#ifndef FLUID_SIMULATOR_CUDA_HELPERS_H
#define FLUID_SIMULATOR_CUDA_HELPERS_H

#define checkCudaErrors(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Launches a CUDA kernel, check for errors, and synchronize the device.
#define launchKernel(kernelCall) { \
    kernelCall; \
    cudaError_t errSync = cudaDeviceSynchronize(); \
    if (errSync != cudaSuccess) { \
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(errSync)); \
        exit(EXIT_FAILURE); \
    } \
    cudaError_t errAsync = cudaGetLastError(); \
    if (errAsync != cudaSuccess) { \
        fprintf(stderr, "CUDA async error: %s\n", cudaGetErrorString(errAsync)); \
        exit(EXIT_FAILURE); \
    } \
}


// Times function execution time and prints
#define timeKernelExecution(kernelCall, kernelName) { \
    cudaEvent_t __timeKernelExecution_startEvent, __timeKernelExecution_stopEvent; \
    checkCudaErrors(cudaEventCreate(&__timeKernelExecution_startEvent)); \
    checkCudaErrors(cudaEventCreate(&__timeKernelExecution_stopEvent)); \
    checkCudaErrors(cudaEventRecord(__timeKernelExecution_startEvent)); \
    kernelCall; \
    checkCudaErrors(cudaEventRecord(__timeKernelExecution_stopEvent)); \
    checkCudaErrors(cudaEventSynchronize(__timeKernelExecution_stopEvent)); \
    float __timeKernelExecution_elapsedMs = 0.0f; \
    checkCudaErrors(cudaEventElapsedTime(&__timeKernelExecution_elapsedMs, __timeKernelExecution_startEvent, __timeKernelExecution_stopEvent)); \
    printf("Kernel %s executed in %.2f ms\n", kernelName, __timeKernelExecution_elapsedMs); \
    checkCudaErrors(cudaEventDestroy(__timeKernelExecution_startEvent)); \
    checkCudaErrors(cudaEventDestroy(__timeKernelExecution_stopEvent)); \
}


#endif //FLUID_SIMULATOR_CUDA_HELPERS_H

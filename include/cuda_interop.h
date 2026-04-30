/** @file cuda_interop.h
 * @brief Header file for CUDA interop functions.
 *
 * This file declares functions for initializing and releasing the CUDA runtime,
 * as well as running a simulation step using CUDA.
 */


#ifndef CUDA_INTEROP_H
#define CUDA_INTEROP_H

#include "data_structures.h" // For CudaData, GLData, ParticleSystem

void cuda_init_runtime(); // Renamed from initCUDA
void cuda_release_runtime(); // Renamed from releaseCUDA
void cuda_run_simulation_step(); // This will orchestrate mapping, kernel, unmapping


#endif // CUDA_INTEROP_H
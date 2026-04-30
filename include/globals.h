#ifndef GLOBALS_H
#define GLOBALS_H

#include "data_structures.h"

// External declarations of global variables
extern ParticleSystem* gd_particleSystem; // Device pointer for ParticleSystem

extern FluidGrid*     gd_fluidGrid; // Device pointer for FluidGrid

extern CudaData       g_cudaData; // Global CUDA data structure for interop
extern GLData         g_glData; // Global OpenGL data structure for rendering




#endif // GLOBALS_H
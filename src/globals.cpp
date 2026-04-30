#include "globals.h"

// Definitions of global variables
ParticleSystem gh_particleSystem;
ParticleSystem* gd_particleSystem;
FluidGrid      gh_fluidGrid;
FluidGrid*     gd_fluidGrid;
CudaData       g_cudaData;
GLData         g_glData;

int* d_cell_heads = nullptr; // Device pointer for particle keys
int* d_particle_next = nullptr; // Device pointer for particle indices
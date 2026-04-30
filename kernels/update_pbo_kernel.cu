#include "cuda_kernels.cuh"

__global__ void update_pbo_kernel(ParticleSystem* d_ps, float* pbo_mapped_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_ps->numParticles) {
        pbo_mapped_ptr[idx * 3 + 0] = d_ps->x[idx];
        pbo_mapped_ptr[idx * 3 + 1] = d_ps->y[idx];
        pbo_mapped_ptr[idx * 3 + 2] = d_ps->z[idx];
    }
}
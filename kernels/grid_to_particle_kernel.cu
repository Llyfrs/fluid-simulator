
#include "cuda_kernels.cuh"

template<GridType type>
__device__ float trilinear_interpolate(
        float gx, float gy, float gz,
        const float* __restrict__ grid_array,
        FluidGrid* d_fg)
{
    float shift_x, shift_y, shift_z;
    int dim_x, dim_y, dim_z;
    const int nx = d_fg->nx, ny = d_fg->ny, nz = d_fg->nz;

    if constexpr (type == U_GRID) {
        shift_x = 0.f;  shift_y = -0.5f; shift_z = -0.5f;
        dim_x = nx + 1; dim_y = ny;      dim_z = nz;
    } else if constexpr (type == V_GRID) {
        shift_x = -0.5f; shift_y = 0.f;  shift_z = -0.5f;
        dim_x = nx;      dim_y = ny + 1; dim_z = nz;
    } else {
        shift_x = -0.5f; shift_y = -0.5f; shift_z = 0.f;
        dim_x = nx;      dim_y = ny;      dim_z = nz + 1;
    }

    float x = gx + shift_x, y = gy + shift_y, z = gz + shift_z;

    int i = max(0, min((int)floorf(x), dim_x - 2));
    int j = max(0, min((int)floorf(y), dim_y - 2));
    int k = max(0, min((int)floorf(z), dim_z - 2));

    float fx = x - i, fy = y - j, fz = z - k;

    int idx000, idx100, idx010, idx110, idx001, idx101, idx011, idx111;

    if constexpr (type == U_GRID) {
        idx000 = u_idx(i, j, k, nx, ny, nz); idx100 = u_idx(i+1, j, k, nx, ny, nz);
        idx010 = u_idx(i, j+1, k, nx, ny, nz); idx110 = u_idx(i+1, j+1, k, nx, ny, nz);
        idx001 = u_idx(i, j, k+1, nx, ny, nz); idx101 = u_idx(i+1, j, k+1, nx, ny, nz);
        idx011 = u_idx(i, j+1, k+1, nx, ny, nz); idx111 = u_idx(i+1, j+1, k+1, nx, ny, nz);
    } else if constexpr (type == V_GRID) {
        idx000 = v_idx(i, j, k, nx, ny, nz); idx100 = v_idx(i+1, j, k, nx, ny, nz);
        idx010 = v_idx(i, j+1, k, nx, ny, nz); idx110 = v_idx(i+1, j+1, k, nx, ny, nz);
        idx001 = v_idx(i, j, k+1, nx, ny, nz); idx101 = v_idx(i+1, j, k+1, nx, ny, nz);
        idx011 = v_idx(i, j+1, k+1, nx, ny, nz); idx111 = v_idx(i+1, j+1, k+1, nx, ny, nz);
    } else {
        idx000 = w_idx(i, j, k, nx, ny, nz); idx100 = w_idx(i+1, j, k, nx, ny, nz);
        idx010 = w_idx(i, j+1, k, nx, ny, nz); idx110 = w_idx(i+1, j+1, k, nx, ny, nz);
        idx001 = w_idx(i, j, k+1, nx, ny, nz); idx101 = w_idx(i+1, j, k+1, nx, ny, nz);
        idx011 = w_idx(i, j+1, k+1, nx, ny, nz); idx111 = w_idx(i+1, j+1, k+1, nx, ny, nz);
    }

    float v000 = grid_array[idx000], v100 = grid_array[idx100];
    float v010 = grid_array[idx010], v110 = grid_array[idx110];
    float v001 = grid_array[idx001], v101 = grid_array[idx101];
    float v011 = grid_array[idx011], v111 = grid_array[idx111];

    float v00 = v000 + fx * (v100 - v000);
    float v01 = v001 + fx * (v101 - v001);
    float v10 = v010 + fx * (v110 - v010);
    float v11 = v011 + fx * (v111 - v011);

    float v0 = v00 + fy * (v10 - v00);
    float v1 = v01 + fy * (v11 - v01);

    return v0 + fz * (v1 - v0);
}


// Before optimization grid_to_particle executed in 1.34 ms
// After optimization grid_to_particle executed in 0.15 ms

/**
 * @brief Updates particle velocities from the grid using a PIC/FLIP blend.
 *
 * @param d_ps      Pointer to the particle system.
 * @param d_fg      Pointer to the fluid grid.
 * @param alpha     The PIC/FLIP blend factor (e.g., 0.1 for 10% PIC).
 */
__global__ void grid_to_particle_kernel(ParticleSystem* d_ps, FluidGrid* d_fg, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_ps->numParticles) return;

    // 1. Get particle position
    float px = d_ps->x[idx];
    float py = d_ps->y[idx];
    float pz = d_ps->z[idx];

    // 2. Convert to normalized grid coordinates
    const float inv_dx = 1.0f / d_fg->grid_spacing;
    float gx = (px - d_fg->originX) * inv_dx;
    float gy = (py - d_fg->originY) * inv_dx;
    float gz = (pz - d_fg->originZ) * inv_dx;

    // 3. Interpolate velocities from both the OLD and NEW grid fields
    //    (This requires a proper trilinear_interpolate helper function)
    float u_old = trilinear_interpolate<U_GRID>(gx, gy, gz, d_fg->u_temp, d_fg);
    float v_old = trilinear_interpolate<V_GRID>(gx, gy, gz, d_fg->v_temp, d_fg);
    float w_old = trilinear_interpolate<W_GRID>(gx, gy, gz, d_fg->w_temp, d_fg);

    float u_new = trilinear_interpolate<U_GRID>(gx, gy, gz, d_fg->u, d_fg);
    float v_new = trilinear_interpolate<V_GRID>(gx, gy, gz, d_fg->v, d_fg);
    float w_new = trilinear_interpolate<W_GRID>(gx, gy, gz, d_fg->w, d_fg);


    // 4. Calculate PIC and FLIP velocities
    // PIC velocity is simply the new grid velocity
    float u_pic = u_new;
    float v_pic = v_new;
    float w_pic = w_new;

    // FLIP velocity is the particle's old velocity plus the CHANGE from the grid
    float u_flip = d_ps->vx[idx] + (u_new - u_old);
    float v_flip = d_ps->vy[idx] + (v_new - v_old);
    float w_flip = d_ps->vz[idx] + (w_new - w_old);

    // 5. Blend them to get the final new velocity for the particle
    d_ps->vx[idx] = (1.0f - alpha) * u_flip + alpha * u_pic;
    d_ps->vy[idx] = (1.0f - alpha) * v_flip + alpha * v_pic;
    d_ps->vz[idx] = (1.0f - alpha) * w_flip + alpha * w_pic;

}
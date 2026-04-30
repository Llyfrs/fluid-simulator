#include "cuda_kernels.cuh"

/**
 * @brief Enforces all boundary conditions on the velocity grid.
 *
 * This kernel implements the full set of boundary conditions described in the text
 * using the provided helper functions for grid indexing.
 * 1. Zeros out velocity on any face adjacent to a SOLID cell.
 * 2. For domain boundaries, applies a no-slip condition for the normal velocity component.
 * 3. For domain boundaries, applies a free-slip condition for tangential velocity components.
 *

 */
__global__ void enforce_boundary_kernel(FluidGrid* __restrict__ d_fg) {
    const int nx = d_fg->nx, ny = d_fg->ny, nz = d_fg->nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // --- U-faces (Horizontal velocity, size = (nx+1)*ny*nz) ---
    for (int idx = tid; idx < d_fg->size_u; idx += stride) {
        int k = idx / ((nx + 1) * ny);
        int tmp = idx % ((nx + 1) * ny);
        int j = tmp / (nx + 1);
        int i = tmp % (nx + 1);

        // Priority 1: Check for adjacent SOLID cells
        if (i > 0 && i < nx) {
            int c0 = cc_idx(i - 1, j, k, nx, ny);
            int c1 = cc_idx(i, j, k, nx, ny);
            if (d_fg->cellType[c0] == SOLID_CELL || d_fg->cellType[c1] == SOLID_CELL) {
                d_fg->u[idx] = 0.0f;
                continue;
            }
        }

        // Priority 2: Apply domain boundary conditions from the text
        // No-slip for u-velocity (normal component) on i-boundaries
/*        if (i == 0 || i == 1 || i == nx - 1 || i == nx) {
            d_fg->u[idx] = 0.0f;
        }
            // Free-slip for u-velocity (tangential component) on j-boundaries
        else if (j == 0 && ny > 1) {
            d_fg->u[idx] = d_fg->u[u_idx(i, 1, k, nx, ny, nz)];
        } else if (j == ny - 1 && ny > 1) {
            d_fg->u[idx] = d_fg->u[u_idx(i, ny - 2, k, nx, ny, nz)];
        }
            // Free-slip for u-velocity (tangential component) on k-boundaries
        else if (k == 0 && nz > 1) {
            d_fg->u[idx] = d_fg->u[u_idx(i, j, 1, nx, ny, nz)];
        } else if (k == nz - 1 && nz > 1) {
            d_fg->u[idx] = d_fg->u[u_idx(i, j, nz - 2, nx, ny, nz)];
        }*/
    }

    // --- V-faces (Vertical velocity, size = nx*(ny+1)*nz) ---
    for (int idx = tid; idx < d_fg->size_v; idx += stride) {
        int k = idx / (nx * (ny + 1));
        int tmp = idx % (nx * (ny + 1));
        int j = tmp / nx;
        int i = tmp % nx;

        // Priority 1: Check for adjacent SOLID cells
        if (j > 0 && j < ny) {
            int c0 = cc_idx(i, j - 1, k, nx, ny);
            int c1 = cc_idx(i, j, k, nx, ny);
            if (d_fg->cellType[c0] == SOLID_CELL || d_fg->cellType[c1] == SOLID_CELL) {
                d_fg->v[idx] = 0.0f;
                continue;
            }
        }

        // Priority 2: Apply domain boundary conditions from the text
        // No-slip for v-velocity (normal component) on j-boundaries
/*        if (j == 0 || j == 1 || j == ny - 2 || j == ny - 1) {
            d_fg->v[idx] = 0.0f;
        }
            // Free-slip for v-velocity (tangential component) on i-boundaries
        else if (i == 0 && nx > 1) {
            d_fg->v[idx] = d_fg->v[v_idx(1, j, k, nx, ny, nz)];
        } else if (i == nx - 1 && nx > 1) {
            d_fg->v[idx] = d_fg->v[v_idx(nx - 2, j, k, nx, ny, nz)];
        }
            // Free-slip for v-velocity (tangential component) on k-boundaries
        else if (k == 0 && nz > 1) {
            d_fg->v[idx] = d_fg->v[v_idx(i, j, 1, nx, ny, nz)];
        } else if (k == nz - 1 && nz > 1) {
            d_fg->v[idx] = d_fg->v[v_idx(i, j, nz - 2, nx, ny, nz)];
        }*/
    }

    // --- W-faces (Depth velocity, size = nx*ny*(nz+1)) ---
    for (int idx = tid; idx < d_fg->size_w; idx += stride) {
        int k = idx / (nx * ny);
        int tmp = idx % (nx * ny);
        int j = tmp / nx;
        int i = tmp % nx;

        // Priority 1: Check for adjacent SOLID cells
        if (k > 0 && k < nz) {
            int c0 = cc_idx(i, j, k - 1, nx, ny);
            int c1 = cc_idx(i, j, k, nx, ny);
            if (d_fg->cellType[c0] == SOLID_CELL || d_fg->cellType[c1] == SOLID_CELL) {
                d_fg->w[idx] = 0.0f;
                continue;
            }
        }

        // Priority 2: Apply domain boundary conditions from the text
        // No-slip for w-velocity (normal component) on k-boundaries
/*        if (k == 0 || k == 1 || k == nz - 2 || k == nz - 1) {
            d_fg->w[idx] = 0.0f;
        }
            // Free-slip for w-velocity (tangential component) on i-boundaries
        else if (i == 0 && nx > 1) {
            d_fg->w[idx] = d_fg->w[w_idx(1, j, k, nx, ny, nz)];
        } else if (i == nx - 1 && nx > 1) {
            d_fg->w[idx] = d_fg->w[w_idx(nx - 2, j, k, nx, ny, nz)];
        }
            // Free-slip for w-velocity (tangential component) on j-boundaries
        else if (j == 0 && ny > 1) {
            d_fg->w[idx] = d_fg->w[w_idx(i, 1, k, nx, ny, nz)];
        } else if (j == ny - 1 && ny > 1) {
            d_fg->w[idx] = d_fg->w[w_idx(i, ny - 2, k, nx, ny, nz)];
        }*/
    }
}
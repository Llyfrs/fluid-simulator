/** * @file fluid_grid.h
 * @brief Header file for FluidGrid data structure and related functions.
 *
 * This file declares functions for allocating, freeing, and managing the FluidGrid data structure,
 * which is used to store fluid simulation data on the GPU.
 */

#ifndef FLUID_GRID_H
#define FLUID_GRID_H

#include "data_structures.h"


/** @brief Allocates memory for a FluidGrid on the device.
 *
 * This function allocates memory for the FluidGrid structure and its internal arrays on the GPU.
 * It initializes the grid dimensions, spacing, and origin.
 *
 * @param nx Number of grid cells in the x dimension.
 * @param ny Number of grid cells in the y dimension.
 * @param nz Number of grid cells in the z dimension.
 * @param dx Grid spacing in each dimension.
 * @param originX Origin coordinate in the x dimension.
 * @param originY Origin coordinate in the y dimension.
 * @param originZ Origin coordinate in the z dimension.
 * @return Pointer to the allocated FluidGrid on the device.
 */
FluidGrid* fg_allocate_device(int nx, int ny, int nz, float dx, float originX, float originY, float originZ);

/** @brief Frees memory allocated by fg_allocate_device on the device.
 *
 * This function frees all memory associated with the FluidGrid structure.
 * @param fg Reference to the FluidGrid structure to be freed.
 */
void fg_freeMem_device(FluidGrid* d_fg);


/** @brief Swaps the pressure and temporary pressure pointers in the FluidGrid.
 *
 * This function swaps the pointers for pressure and temporary pressure in the FluidGrid structure.
 * It's used after jacobi iteration to apply the new pressure values.
 *
 * @param d_fg Pointer to the FluidGrid on the device.
 */
void fg_swap_pressure_pointers(FluidGrid* d_fg);

void fg_save_velocity(FluidGrid* d_grid); // Not used in the end, slower that kernel copy

#endif // FLUID_GRID_H
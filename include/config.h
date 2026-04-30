/** * @file config.h
 * @brief Configuration file for the fluid simulation.
 *
 * This file contains constants and configuration parameters used throughout the fluid simulation.
 * It defines simulation time step, grid dimensions, particle spacing, and other important parameters.
 */

#ifndef CONFIG_H
#define CONFIG_H



#include <cuda_runtime.h>
#include <cmath>

inline constexpr float SIMULATION_TIME_STEP = 0.00001f;
inline constexpr float SIMULATIONS_PER_FRAME = 1;

inline constexpr int NUM_JACOBI_ITERATIONS = 10;
inline constexpr int NUM_THREADS = 512;

inline constexpr float GRAVITY = -2.65f;
inline constexpr float BLEND = 0.05f; // 0 == Fully FLIP, if 1 == Fully PIC between is blend

inline constexpr int GRID_NX = 32;
inline constexpr int GRID_NY = 50;
inline constexpr int GRID_NZ = 40;

inline constexpr float GRID_SPACING = 0.08f;

inline float3 GRID_ORIGIN = make_float3(
        -GRID_NX * GRID_SPACING * 0.5f,
        -GRID_NY * GRID_SPACING * 0.5f,
        -GRID_NZ * GRID_SPACING * 0.5f
);

inline float3 FLUID_MIN_BOUNDS = make_float3(-1.0f, -1.5f, -1.5f);
inline float3 FLUID_MAX_BOUNDS = make_float3(0.5f, 1.8f, 0.5f);

inline int PARTICLES_PER_CELL_AXIS = 3; // Number of particles per cell along each axis (2x2x2 = 8 particles per cell)

inline float PARTICLE_SPACING = GRID_SPACING / PARTICLES_PER_CELL_AXIS;

inline float3 FLUID_SIZE = make_float3(
        FLUID_MAX_BOUNDS.x - FLUID_MIN_BOUNDS.x,
        FLUID_MAX_BOUNDS.y - FLUID_MIN_BOUNDS.y,
        FLUID_MAX_BOUNDS.z - FLUID_MIN_BOUNDS.z
        );

inline int3 PARTICLES_PER_DIMENSION = make_int3(
        floorf(FLUID_SIZE.x / PARTICLE_SPACING),
        floorf(FLUID_SIZE.y / PARTICLE_SPACING),
        floorf(FLUID_SIZE.z / PARTICLE_SPACING)
);

inline int NUM_PARTICLES = PARTICLES_PER_DIMENSION.x * PARTICLES_PER_DIMENSION.y * PARTICLES_PER_DIMENSION.z;

#endif // CONFIG_H

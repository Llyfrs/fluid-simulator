/** * @file data_structures.h
 * @brief Contains data structures used in the fluid simulation.
 *
 * This file defines the ParticleSystem, FluidGrid, CudaData, and GLData structures,
 * which are used to manage particle positions, fluid grid properties, CUDA data,
 * and OpenGL rendering data respectively.
 */
#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#define GLEW_STATIC
#include <GL/glew.h>         // For GLuint in GLData
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // For cudaGraphicsResource in CudaData
#include <cstring>


/** @brief Structure representing a particle system.
 *
 * This structure holds arrays for particle positions and velocities in 3D space,
 * as well as the number of particles in the system.
 */
struct ParticleSystem {
    // Particle positions in 3D space
    float* x; float* y; float* z;

    // Particle velocities in 3D space
    float* vx; float* vy; float* vz;

    int numParticles; // Number of particles in the system

    ParticleSystem() : x(nullptr), y(nullptr), z(nullptr),
                       vx(nullptr), vy(nullptr), vz(nullptr),
                       numParticles(0) {}
};


/** @brief Structure representing a fluid grid for simulation.
 *
 * This structure holds arrays for fluid velocities, pressure, divergence,
 * and other properties used in fluid simulation on a 3D grid.
 */
struct FluidGrid {

    float* u; // Velocity x-axis
    float* v; // Velocity y-axis
    float* w; // Velocity z-axis

    float* u_weights; // Weights for velocity interpolation
    float* v_weights; // Weights for velocity interpolation
    float* w_weights; // Weights for velocity interpolation

    float* u_temp; float* v_temp; float* w_temp; // Temporary storage for velocities during updates

    float* pressure; float* divergence;

    float* temp_pressure; // Temporary storage for pressure during updates

    float* r ; // Temporary storage for residuals during pressure solve
    float* d ; // Temporary storage for direction vectors during pressure solve
    float* q ; // Temporary storage for pressure during pressure solve

    // Cell types: 0 = empty, 1 = fluid, 2 = solid, 3 = boundary
    int* cellType;

    float* phi; // Negative = inside fluid, positive = outside fluid, zero = on the surface

    // Dimensions and grid properties
    int nx, ny, nz;

    // Grid spacing
    float grid_spacing;

    // Origin of the grid in world coordinates
    float originX, originY, originZ;

    // Sizes of the arrays
    size_t size_u, size_v, size_w, size_cellCentered;

    FluidGrid() : u(nullptr), v(nullptr), w(nullptr), pressure(nullptr),
                  divergence(nullptr), cellType(nullptr), u_temp(nullptr),
                  v_temp(nullptr), w_temp(nullptr), phi(nullptr),
                  nx(0), ny(0), nz(0), grid_spacing(0.0f),
                  originX(0.0f), originY(0.0f), originZ(0.0f),
                  size_u(0), size_v(0), size_w(0), size_cellCentered(0) {}
};


/** @brief Structure for managing CUDA data and resources.
 *
 * This structure holds device pointers for particle positions and velocities,
 * as well as a CUDA graphics resource for OpenGL interoperability.
 */
struct CudaData {
    struct cudaGraphicsResource* pboResource;
    CudaData() {
        memset(this, 0, sizeof(CudaData));
    }
};

/** @brief Structure for managing OpenGL data and resources.
 *
 * This structure holds OpenGL buffer IDs, viewport dimensions, camera angles,
 * and shader program ID for rendering the particle system.
 */
struct GLData {
    GLuint pboID;
    unsigned int viewportWidth = 1024;
    unsigned int viewportHeight = 768;
    float cameraAngleXY = 45.0f;  // Start with a 45-degree rotation
    float cameraAngleZ = 30.0f;   // Start with a 30-degree elevation
    float cameraDistance = 5.0f;   // Start 5 units away from origin

    GLuint shaderProgramID; // OpenGL shader program ID
    GLuint vaoID;        // Vertex Array Object ID for rendering

    GLData() : pboID(0) {} // Initialize pboID
};

#endif // DATA_STRUCTURES_H
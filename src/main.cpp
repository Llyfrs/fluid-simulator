#include "globals.h"
#include "config.h"
#include "particle_system.h"
#include "fluid_grid.h"      // If you initialize it
#include "renderer_gl.h"
#include "cuda_interop.h"

#include <cstdio>   // For printf
#include <cstdlib>  // For atexit, srand
#include <ctime>    // For time (if used with srand)

// Forward declaration for atexit
void app_release_all_resources();

int main(int argc, char* argv[]) {
    printf("Starting application...\n");

    srand(time(NULL)); // TODO: Not sure if used anywhere at this point




    printf("Starting Render");

    // 1. Initialize OpenGL and GLUT (creates window and GL context)
    renderer_init_gl(argc, argv);

    printf("Starting Render 2");

    // 2. Prepare OpenGL Objects (VBOs, etc.) - *after* GL context
    renderer_prepare_gl_objects();

    printf("Starting Cuda Init");

    // 3. Initialize CUDA - *after* OpenGL objects for interop
    cuda_init_runtime();

    printf("Setting At Exit");

    // 4. Register clean-up function
    atexit(app_release_all_resources);

    // 5. Start the GLUT main loop
    printf("Entering GLUT main loop...\n");
    glutMainLoop();

    // Usually unreachable
    return 0;
}

void app_release_all_resources() {
    printf("Application exiting, releasing all resources...\n");
    // Release in reverse order of initialization
    cuda_release_runtime();    // Unregister PBO, free CUDA memory
    renderer_release_gl();     // Delete VBOs
    // fg_freeMem(gh_fluidGrid); // If using fluid grid
    printf("All resources released.\n");
}
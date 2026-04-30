cmake_minimum_required(VERSION 3.18)
project(fluid-simulator LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Find OpenGL, GLEW, GLUT
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED) # On Linux, often part of freeglut-dev



# Include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        # Add GLUT include directory if necessary, e.g. /usr/include/GL on some systems
)

include_directories(/opt/cuda/include)

# CMakeLists.txt
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR}/bin)


# --- Define your executable ---
add_executable(fluid_simulator
        src/main.cpp
        src/globals.cpp
        src/particle_system.cpp
        src/renderer_gl.cpp
        src/cuda_interop.cu
        src/fluid_grid.cpp
        kernels/update_particle_positions.cu
        kernels/update_pbo_kernel.cu
        kernels/initialize_particles_kernel.cu
        kernels/particle_to_grid_kernel.cu
        kernels/normalize_grid_velocities_kernel.cu
        kernels/zero_out_fluid_grid_kernel.cu
        kernels/add_gravity_kernel.cu
        kernels/initialize_fluid_grid_kernel.cu
        kernels/enforce_boundary_kernel.cu
        kernels/calculate_divergence_kernel.cu
        kernels/jacobi_iteration_kernel.cu
        kernels/project_velocities_kernel.cu
        kernels/save_velocities_kernel.cu
        kernels/grid_to_particle_kernel.cu
        kernels/alternative_p2g_kernels.cu
        include/cuda_helpers.h
        include/cuda_kernels.cuh
)

# --- Link libraries ---
target_link_libraries(fluid_simulator
        PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_cudart_static_LIBRARY} # Or shared, depending on preference
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${GLUT_LIBRARIES}
)



# If your CUDA files (.cu) include headers that need to be processed by NVCC
# as C++ (e.g. from GLM or other C++ libs), you might need:
set_target_properties(fluid_simulator PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(fluid_simulator PROPERTIES CUDA_SEPARABLE_COMPILATION OFF) # Or ON if needed

# For MSVC, you might need to specify the host compiler for NVCC if it's not auto-detected
# if(MSVC)
#   set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.2X.XXXXX/bin/Hostx64/x64/cl.exe")
# endif()

# Output directory (optional)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
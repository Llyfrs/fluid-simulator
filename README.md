# Fluid Simulation

This is a CUDA + OpenGL fluid/particle simulation experiment, created as a project for the **Parallel Algorithms** university class. I attempted to follow the tutorial [Physics-Based Simulation & Animation of Fluids](https://unusualinsights.github.io/fluid_tutorial/#home) by Chand T. John, Ph.D. Overall, it turned out reasonably well, though the simulation does not seem stable over longer time scales.

## Dependencies

- CMake \(>= 3.18\)
- A C++17 compiler
- CUDA Toolkit \(NVCC\)
- OpenGL + GLEW + GLUT \(on Linux, usually **freeglut**\)

On Arch Linux this is typically covered by packages like `cmake`, `cuda`, `glew`, `freeglut`, and a Mesa/OpenGL driver stack.

## Build

This repo uses CMake. If `cmake` in your PATH is a broken shim, you can call the system binary directly as `/usr/bin/cmake`.

```bash
mkdir -p build
/usr/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build build -j
```

The executable is written to `build/bin/fluid_simulator`. Shaders are copied to `build/bin/shaders/` during the CMake configure step.

## Run

```bash
./build/bin/fluid_simulator
```

If you see missing-shader errors, run from `build/bin/` so relative shader paths resolve:

```bash
cd build/bin
./fluid_simulator
```



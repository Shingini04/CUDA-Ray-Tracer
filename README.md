# CUDA Weekend Ray Tracer

A GPU-accelerated path tracer built with CUDA, based on the ideas from
_Ray Tracing in One Weekend_ and Roger Allen's CUDA adaptation. The project
renders a physically inspired 3D scene made of hundreds of spheres with
diffuse, metallic, and dielectric materials, using parallel GPU kernels to
speed up the ray tracing workload.

This repository is written as both a working CUDA project and a learning
record. It documents how a CPU-style ray tracer can be moved onto the GPU,
what design changes were needed, and the main issues faced while building and
understanding the CUDA version.

## Aim of the Project

The aim of this project was to understand how ray tracing works at a low level
and how the same rendering problem can be accelerated using GPU parallelism.
Ray tracing is computationally expensive because every pixel needs many rays,
and every ray may bounce through the scene multiple times. That makes it a
good project for learning CUDA because each pixel can be computed mostly
independently by a GPU thread.

The project focuses on:

- Building a simple path tracer from mathematical primitives.
- Understanding rays, vectors, cameras, materials, and object intersections.
- Moving rendering work from CPU-style code to CUDA kernels.
- Managing GPU memory and device-side objects.
- Using random sampling for anti-aliasing, diffuse reflection, and materials.
- Producing a final rendered image in PPM format.

## What the Renderer Does

The renderer generates a scene containing many spheres placed across a ground
plane. Each sphere can use one of several material models:

- Lambertian diffuse material for matte surfaces.
- Metal material with controllable fuzziness for reflective surfaces.
- Dielectric material for glass-like refraction.

For every pixel, the CUDA renderer launches GPU threads that trace rays from
the camera into the scene. Rays bounce between objects, scatter depending on
the material they hit, and accumulate color contribution over multiple samples.
The final image is gamma corrected and written as a `.ppm` image.

## Project Progress

The project progressed in stages, starting from a very small CUDA rendering
kernel and gradually growing into a full path tracer.

### 1. Basic CUDA Rendering

The first step was learning the CUDA execution model: launching kernels,
mapping GPU threads to image pixels, and writing color values into a
framebuffer. At this stage, the image generation was simple, but it established
the foundation of dividing work across thousands of GPU threads.

### 2. Vector and Ray Math

Next, the project introduced the mathematical building blocks required for ray
tracing. The `vec3` class handles 3D vector operations such as addition,
subtraction, dot product, cross product, normalization, and scalar operations.
The `ray` class represents rays with an origin and direction.

Because the same vector code is used in both host and device contexts, the
methods needed CUDA annotations such as `__host__` and `__device__`.

### 3. Sphere Intersection

After setting up rays, the next milestone was detecting intersections between
rays and spheres. This required solving the quadratic equation for a ray-sphere
intersection and returning hit information such as the hit point, surface
normal, and distance along the ray.

This was the first point where the renderer started behaving like a real ray
tracer instead of only generating a procedural image.

### 4. Scene Representation on the GPU

The renderer then moved from a single sphere to a list of hittable objects.
This required creating scene objects directly on the GPU and managing pointers
to CUDA-side classes. Files such as `hitable.h`, `hitable_list.h`, and
`sphere.h` define the object abstraction used by the renderer.

This stage was important because normal C++ object ownership does not
automatically work the same way on the GPU. The scene has to be allocated,
constructed, used, and deleted carefully from CUDA kernels.

### 5. Random Sampling with cuRAND

To improve image quality, the project added random sampling. Randomness is used
for anti-aliasing, diffuse scattering, material behavior, and camera depth of
field. CUDA requires each thread to have its own random number state, so the
project uses `curandState` values initialized per pixel.

This was a major step because realistic path tracing depends heavily on random
sampling, but random state must be handled carefully in parallel code.

### 6. Materials and Light Transport

The renderer then added material classes for diffuse, metal, and glass-like
objects. Each material controls how a ray scatters after hitting a surface.
The final color is calculated by following multiple ray bounces through the
scene.

Recursive ray tracing works naturally in CPU code, but recursion is expensive
and risky on the GPU. To avoid GPU stack issues, the recursive color function
was converted into an iterative loop with a maximum bounce depth.

### 7. Camera and Depth of Field

The final stage added a configurable camera with field of view, focus distance,
and aperture. This allows the rendered scene to have a more realistic
perspective and depth-of-field blur.

The final renderer produces the classic random sphere scene from _Ray Tracing
in One Weekend_, accelerated with CUDA.

## Issues Faced

This project involved several practical CUDA challenges:

### CUDA Host and Device Function Boundaries

One of the early challenges was understanding which functions run on the CPU
and which run on the GPU. Regular C++ methods cannot always be called inside a
CUDA kernel unless they are marked correctly. This required adding `__host__`
and `__device__` annotations to shared utility code.

### GPU Memory Management

Managing objects on the GPU was more complex than normal CPU allocation.
Objects like spheres, materials, and the world list need to be created in
device memory and later cleaned up correctly. This made the project useful for
understanding CUDA memory ownership beyond simple arrays.

### Random Numbers in Parallel Code

Generating random numbers on the GPU is different from using a normal CPU
random function. Each CUDA thread needs its own random state to avoid incorrect
or repeated sampling. The project uses cuRAND to initialize and store
per-thread random states.

### Recursion Limitations

The original ray tracing algorithm is naturally recursive because rays bounce
from surface to surface. On the GPU, deep recursion can cause stack problems
and poor performance. The solution was to convert the recursive color
calculation into a loop with a fixed maximum depth.

### Floating-Point Performance

CUDA performance can be hurt if the code accidentally uses double-precision
floating point where single precision is enough. Constants such as `0.5` may be
treated as doubles unless written as `0.5f`. The project required attention to
float constants to avoid unnecessary performance cost.

### Local Hardware Limitation

The code is written for NVIDIA CUDA systems. On a machine without an NVIDIA GPU
or CUDA toolkit installed, the project can be inspected and versioned, but the
actual render cannot be executed locally. A CUDA-capable system with `nvcc` is
required to build and run the renderer.

## Skills Demonstrated

This project demonstrates:

- CUDA C++ programming.
- GPU kernel launches and thread indexing.
- Unified memory and device memory allocation.
- cuRAND random number generation.
- Object-oriented C++ in CUDA device code.
- Ray-sphere intersection mathematics.
- Camera modeling and depth of field.
- Diffuse, reflective, and refractive material simulation.
- Iterative path tracing with multiple ray bounces.
- Debugging CPU-to-GPU porting issues.

## Build and Run

This project requires an NVIDIA GPU and CUDA toolkit.

To build:

```bash
make cudart
```

To render the image:

```bash
make out.ppm
```

The rendered image is written to:

```text
out.ppm
```

## Project Structure

```text
.
├── main.cu          # CUDA kernels, scene creation, rendering loop, output
├── vec3.h           # 3D vector math
├── ray.h            # Ray representation
├── camera.h         # Camera and depth-of-field logic
├── hitable.h        # Base interface for hittable objects
├── hitable_list.h   # Object list used as the scene world
├── sphere.h         # Sphere geometry and intersection logic
├── material.h       # Diffuse, metal, and dielectric materials
├── Makefile         # CUDA build commands
└── README.md        # Project documentation
```

## Resume Summary

Built a CUDA-based path tracer that renders a 3D scene with hundreds of
spheres, multiple material types, anti-aliasing, random sampling, gamma
correction, and depth of field. Ported core ray tracing logic to GPU kernels,
managed device-side objects and memory, used cuRAND for per-thread sampling,
and replaced recursive ray bouncing with an iterative GPU-safe loop.

## Credit

This project follows the learning path of _Ray Tracing in One Weekend_ and is
based on Roger Allen's CUDA adaptation of the book project. I used it as a
hands-on implementation to understand GPU acceleration, CUDA memory management,
and the practical differences between CPU and GPU ray tracing.

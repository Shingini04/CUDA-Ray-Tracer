# CUDA Ray Tracer

CUDA Ray Tracer is a GPU-accelerated path tracing project written in CUDA C++.
It renders a 3D scene made of many spheres with different materials such as
matte diffuse surfaces, reflective metals, and glass-like dielectric objects.
The renderer traces rays from a virtual camera into the scene, lets them bounce
through the world, and calculates the final pixel colors using random sampling.

I built this project to understand two things together: how a ray tracer works
from the ground up, and how the same kind of heavy rendering workload can be
parallelized on a GPU using CUDA. Ray tracing is a very natural fit for GPU
programming because each pixel can be computed independently, but actually
moving the algorithm to CUDA introduces many interesting challenges around
memory, recursion, random numbers, and device-side C++ code.

The final result is a CUDA renderer that produces a `.ppm` image of the classic
random sphere scene inspired by _Ray Tracing in One Weekend_.

## Why I Built This

I wanted a project that was deeper than a normal graphics demo. A ray tracer is
small enough to understand fully, but it still touches many important computer
science and engineering ideas:

- 3D vector math
- camera projection
- geometry intersection
- materials and light scattering
- random sampling
- performance thinking
- GPU parallelism
- memory management

The goal was not only to get an image on the screen. The bigger goal was to
understand what happens behind that image: how rays are generated, how they hit
objects, how colors are accumulated, and why GPUs are useful for this kind of
work.

This project also helped me become more comfortable reading and adapting
systems-level C++/CUDA code. A lot of the learning came from small details:
where a function is allowed to run, where memory lives, when a pointer points
to host memory versus device memory, and why code that looks valid in normal
C++ may behave differently inside a CUDA kernel.

## Project Aim

The aim of CUDA Ray Tracer is to implement a path tracer that runs its main
rendering work on the GPU.

More specifically, the project aims to:

- Generate rays from a virtual camera for every pixel.
- Detect ray-object intersections with spheres.
- Support multiple material types.
- Simulate diffuse reflection, metallic reflection, and glass refraction.
- Use random sampling for anti-aliasing and softer visual results.
- Trace multiple ray bounces to approximate indirect lighting.
- Run the pixel rendering workload in parallel using CUDA kernels.
- Write the final rendered image as a PPM file.

From a learning point of view, the project is about understanding the journey
from a CPU-style ray tracer to a GPU-oriented implementation.

## What the Project Does

The renderer creates a 3D scene containing a large ground sphere and many
smaller randomly placed spheres. Each object is assigned a material. Some
objects look matte, some look metallic, and some behave like glass.

For each pixel, the renderer:

1. Creates a camera ray.
2. Sends the ray into the scene.
3. Checks which object the ray hits first.
4. Calculates how the ray scatters based on the material.
5. Repeats the process for multiple bounces.
6. Accumulates the color contribution.
7. Averages multiple samples per pixel.
8. Applies gamma correction.
9. Writes the final color to the framebuffer.

Because every pixel can be rendered separately, CUDA is used to assign pixel
work to GPU threads. This allows the renderer to work on many pixels at the
same time instead of processing the image one pixel at a time on the CPU.

## How the Project Progressed

The project grew step by step. Each stage added one important part of the final
renderer.

### 1. Starting with CUDA Kernels

The first step was learning how to launch a CUDA kernel and map GPU threads to
pixels in an image. At this point, the program did not need complex ray tracing
logic. The main goal was to understand how a GPU thread can calculate one part
of the output image.

This stage introduced the basic CUDA workflow:

- allocate memory for the image
- launch a kernel
- calculate pixel coordinates from thread and block indices
- write color data into a framebuffer
- copy or access the result from the host side

This was the foundation for the whole project. Once every thread could be
responsible for one pixel, the rest of the ray tracing logic could be built on
top of that.

### 2. Building the Math Layer

Ray tracing depends heavily on vector math, so the next step was implementing
the basic math structures. The `vec3` class handles operations like addition,
subtraction, multiplication, normalization, dot product, and cross product.

The `ray` class stores the origin and direction of a ray. A ray can be thought
of as a line starting at the camera and moving into the scene. Every visible
pixel begins as one or more rays.

One CUDA-specific challenge appeared here: some functions had to run on both
the CPU and the GPU. That meant methods needed CUDA annotations such as
`__host__` and `__device__`. This was one of the first reminders that CUDA C++
looks like normal C++, but the execution model is very different.

### 3. Adding Sphere Intersections

After rays were working, the next goal was to make them hit something. The
project started with spheres because they are mathematically clean and common
in beginner ray tracers.

Ray-sphere intersection is solved using a quadratic equation. If the equation
has a valid solution, the ray hits the sphere. The renderer then stores useful
information about the hit:

- where the ray hit
- how far along the ray the hit occurred
- the surface normal at the hit point
- the material of the object

This stage made the project feel like a real ray tracer for the first time.
Instead of just producing a color gradient, the renderer could now understand
objects in 3D space.

### 4. Creating a Scene

Once a single sphere worked, the next step was creating a full scene. The
project uses a list of hittable objects, where each object can be tested for
intersection with a ray.

The important files for this part are:

- `hitable.h`
- `hitable_list.h`
- `sphere.h`

This stage was more difficult in CUDA than it would be in normal C++. The
objects are used by GPU kernels, so they need to exist in GPU-accessible
memory. That means the scene cannot simply be created like a normal CPU data
structure and used directly by device code. The project creates the world on
the device and later deletes it properly.

This helped me understand device-side object construction and why GPU memory
management requires careful thinking.

### 5. Adding Random Sampling

A clean ray traced image needs more than one ray per pixel. Multiple samples
help reduce jagged edges and create more realistic material behavior. Random
sampling is also needed for diffuse reflection and depth of field.

The project uses cuRAND for random number generation on the GPU. This was an
important part of the project because random numbers in parallel programs are
not as simple as calling one global random function. Each CUDA thread needs its
own random state.

The renderer initializes random states for pixels and uses them during
rendering. This makes every pixel sample slightly different ray directions,
which improves the final image.

### 6. Implementing Materials

The renderer supports three main material types.

Lambertian materials are diffuse and matte. They scatter light in random
directions, which gives objects a soft non-reflective appearance.

Metal materials reflect rays. A fuzziness value controls how sharp or blurry
the reflection is.

Dielectric materials simulate transparent glass-like surfaces. They can refract
rays through the object and also reflect rays depending on the viewing angle.

This part of the project made the image much more interesting because the same
geometry could now look completely different depending on the material assigned
to it.

### 7. Handling Ray Bounces

In a path tracer, a ray may hit an object, scatter, hit another object, scatter
again, and continue for several bounces. This is how the renderer approximates
light transport.

The original CPU-style algorithm is naturally recursive: a color function calls
itself for every new scattered ray. On the GPU, deep recursion can cause stack
issues and performance problems. To make the implementation more GPU-friendly,
the ray bounce logic is written as a loop with a maximum depth.

This was one of the most important changes in the project. It shows how an
algorithm sometimes has to be reshaped when moving from CPU programming to GPU
programming.

### 8. Adding Camera Effects

The final renderer includes a camera system with field of view, camera
position, focus distance, and aperture. This allows the scene to have
perspective and depth of field.

Depth of field makes objects near the focus distance appear sharper while
objects outside that range become blurred. This gives the final image a more
photographic feel.

## Issues and Challenges Faced

This project was useful because it was not just about writing code that works.
It also exposed several practical problems that come up in GPU programming.

### Host Code vs Device Code

One of the first confusing parts was understanding where each function runs.
Some code runs on the CPU, some code runs on the GPU, and some helper functions
need to be available in both places.

For example, vector operations are used in device kernels, so they must be
compiled for the GPU. Without the correct CUDA annotations, code that looks
fine in normal C++ can fail when called from a kernel.

This taught me to think more carefully about execution location.

### GPU Memory Management

Managing memory was one of the most important challenges. In normal C++, it is
easy to create objects on the heap and pass pointers around. In CUDA, the
renderer has to consider whether the memory is on the host or the device.

The scene objects, object list, materials, and camera need to be accessible
from GPU kernels. This required allocating and constructing objects in a way
that CUDA device code can use.

The project made me more comfortable with the difference between CPU memory,
device memory, and unified memory.

### Random Numbers on the GPU

Random sampling is central to path tracing, but random numbers are more complex
in GPU code. Thousands of threads may run at the same time, and each thread
needs an independent random sequence.

Using cuRAND helped solve this, but it also required setting up a `curandState`
for each pixel. This was a good lesson in how parallel code often needs extra
state management compared to single-threaded code.

### Recursion and GPU Stack Limits

The CPU version of a ray tracer can be written recursively in a very natural
way. CUDA device code can support recursion in some cases, but it is not always
the best choice, especially when many threads are running and each ray may
bounce many times.

The project avoids this by using an iterative loop for ray bounces. This makes
the renderer more predictable and avoids stack overflows from deeply recursive
ray paths.

### Floating-Point Precision

Another issue was floating-point performance. In CUDA code, accidentally using
double-precision constants can slow things down. For example, writing `0.5`
instead of `0.5f` may cause unnecessary double-precision operations.

This made me more aware of how small details in numerical code can affect
performance.

### Hardware Limitation

The project requires an NVIDIA CUDA environment to build and render. Since my
local machine does not have an NVIDIA CUDA setup, I could prepare, inspect, and
version the project locally, but actual rendering requires a CUDA-capable
machine with `nvcc`.

This is also a real-world lesson: GPU projects are tied closely to hardware and
toolchain availability.

## What I Learned

This project helped me connect computer graphics theory with GPU programming.
The most valuable learning was seeing how a simple visual result depends on
many smaller systems working together:

- rays generated from a camera
- geometry intersection
- surface normals
- material scattering
- random sampling
- memory allocation
- GPU kernel execution
- final image output

I also learned that accelerating code with CUDA is not just about moving code
from CPU to GPU. The algorithm often needs to be adjusted to match the GPU
execution model. Recursion, object ownership, random state, and memory access
all need to be considered carefully.

## Skills Demonstrated

This project demonstrates practical experience with:

- CUDA C++
- GPU kernel programming
- thread and block indexing
- unified memory
- device memory allocation
- cuRAND random number generation
- object-oriented C++ in CUDA device code
- ray tracing mathematics
- sphere intersection logic
- path tracing and ray bouncing
- material modeling
- camera modeling
- anti-aliasing
- depth of field
- gamma correction
- debugging CUDA-specific issues

## Build and Run

This project requires:

- NVIDIA GPU
- CUDA toolkit
- `nvcc`
- `make`

Build the renderer:

```bash
make cudart
```

Render the image:

```bash
make out.ppm
```

The output image will be generated as:

```text
out.ppm
```

## Project Structure

```text
.
├── main.cu          # CUDA kernels, scene setup, rendering, cleanup, output
├── vec3.h           # 3D vector math
├── ray.h            # Ray class
├── camera.h         # Camera and depth-of-field logic
├── hitable.h        # Base interface for objects that rays can hit
├── hitable_list.h   # Scene object list
├── sphere.h         # Sphere geometry and intersection logic
├── material.h       # Lambertian, metal, and dielectric materials
├── Makefile         # CUDA build commands
└── README.md        # Project explanation and documentation
```

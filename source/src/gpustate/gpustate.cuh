#ifndef GPUSTATE_HPP
#define GPUSTATE_HPP

#include <cuda_gl_interop.h>
#include <helper_math.h>
#include <stdio.h>

// Macro to handle checking all the possible cuda errors
#define CUDAERR(ans)                                                           \
  { exit_if_cuda_err((ans), __FILE__, __LINE__); }
inline void exit_if_cuda_err(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDAERR: %s ||| %s:%d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

// Same struct as on CPU-side, but using float3.  Since float3 and
// glm::vec3 are memory-compatible, the entire structs will be
// memory-compatible. Using float3 while doing work in CUDA is faster
// though, hence this struct.
struct CUParticle {
  float3 pos;
  float3 velocity;
  char type;
};

// Class containing the allocations of the CUDA device, necessary for
// the kernels to have something to run on.
struct GPUState {
  CUParticle *particles = 0;
  float3 *velocities = 0;
  struct cudaGraphicsResource *resources = 0;
  void *glptr = 0;

  // We init worldstate pretty long before cuda is up, so use separate methods

  void init(const CUParticle *in, size_t n, GLuint vbo_sphere_inst);
  void clean();
};

__global__ void update_GL(CUParticle *particles, void *glptr, size_t n);

#endif

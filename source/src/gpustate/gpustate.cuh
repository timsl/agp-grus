#ifndef GPUSTATE_HPP
#define GPUSTATE_HPP

#include <common.hpp>
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

struct CUParticle {
  float3 pos;
  float3 velocity;
  char type;
};

struct GPUState {
  CUParticle *particles = 0;
  float3 *velocities = 0;
  struct cudaGraphicsResource *resources = 0;
  void *glptr = 0;

  // We init worldstate pretty long before cuda is up, so use separate methods

  void init(const CUParticle *in, size_t n, GLuint vbo_sphere_inst);
  void clean();
};

#endif
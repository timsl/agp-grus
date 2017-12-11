#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <helper_math.h>
#include "state.hpp"

// Macro to handle checking all the possible cuda errors
#define CUDAERR(ans) { exit_if_cuda_err((ans), __FILE__, __LINE__); }
inline void exit_if_cuda_err(cudaError_t code, const char *file, int line){
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDAERR: %s ||| %s:%d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

struct CUParticle {
  float3 pos;
  float3 velocity;
  char type;
};

void update(WorldState *world, float dt);

// We need to separate these kernels since they should seem to be
// concurrent, and therefore we can't update the velocity beforehand
// since there is a check for velocity in the if cases.
__global__ void calculate_velocities(const CUParticle *particles, float3 *velocities,
                                     size_t n, float dt);

__global__ void apply_velocities(CUParticle *particles, float3 *velocities,
                                 size_t n, float dt);

#endif

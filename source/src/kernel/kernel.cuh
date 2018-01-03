#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "common.hpp"
#include "gpustate.cuh"
#include "state.hpp"
#include <helper_math.h>

void update(WorldState *world, float dt);
void first_update(WorldState *world, float dt);

// We need to separate these kernels since they should seem to be
// concurrent, and therefore we can't update the velocity beforehand
// since there is a check for velocity in the if cases.
__global__ void calculate_forces(const CUParticle *particles,
                                 float3 *velocities, size_t n, float dt);

__global__ void apply_forces(CUParticle *particles, float3 *velocities,
                             size_t n, float dt);

__global__ void first_apply_forces(CUParticle *particles, float3 *velocities,
                                   size_t n, float dt);

#endif

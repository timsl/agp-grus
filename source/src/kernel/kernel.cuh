#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "common.hpp"
#include "gpustate.cuh"
#include "state.hpp"
#include <helper_math.h>

// Run the kernels, updating each of the particles through one leapfrog step.
void update(WorldState *world, float dt);
// Same, but the first half-step of leapfrog.
void first_update(WorldState *world, float dt);

// We need to separate these kernels since they should seem to be
// concurrent, and therefore we can't update the velocity beforehand
// since there is a check for velocity in the if cases.

// Calculate the force that 'pi' is affected by from 'pj'.
// Includes the direction of the force.
__device__ float3 body_body_interaction(CUParticle pi, CUParticle pj);

// Calculate all the body-body force interactions, and put them in *forces
__global__ void calculate_forces(const CUParticle *particles,
                                 float3 *forces, size_t n, float dt);

// Apply the forces using the standard leapfrog steps.
// Calculates the new velocity and then immediately the next position.
__global__ void apply_forces(CUParticle *particles, float3 *forces,
                             size_t n, float dt);

// Apply the forces using the first half-step of the leapfrog algorithm.
__global__ void first_apply_forces(CUParticle *particles, float3 *forces,
                                   size_t n, float dt);

#endif

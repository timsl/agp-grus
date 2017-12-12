#include "gpustate.cuh"

void GPUState::init(const CUParticle *in, size_t n) {
  CUDAERR(cudaMalloc(&particles, n * sizeof(*particles)));
  CUDAERR(cudaMalloc(&velocities, n * sizeof(*velocities)));

  // glm is memory compatible with float3 so a struct should be aswell
  CUDAERR(cudaMemcpy(particles, in, n * sizeof(*particles),
                     cudaMemcpyHostToDevice));
}
void GPUState::clean() {
  CUDAERR(cudaFree(particles));
  CUDAERR(cudaFree(velocities));
}
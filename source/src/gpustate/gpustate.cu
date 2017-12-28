#include "gpustate.cuh"

void GPUState::init(const CUParticle *in, size_t n, GLuint vbo_sphere_inst) {
  CUDAERR(cudaMalloc(&particles, n * sizeof(*particles)));
  CUDAERR(cudaMalloc(&velocities, n * sizeof(*velocities)));

  // glm is memory compatible with float3 so a struct should be aswell
  CUDAERR(cudaMemcpy(particles, in, n * sizeof(*particles),
                     cudaMemcpyHostToDevice));

  CUDAERR(cudaGraphicsGLRegisterBuffer(&resources, vbo_sphere_inst,
                                       cudaGraphicsMapFlagsNone));
  CUDAERR(cudaGraphicsMapResources(1, &resources));
  size_t size = 0;
  CUDAERR(cudaGraphicsResourceGetMappedPointer(&glptr, &size, resources));
}

void GPUState::clean() {
  CUDAERR(cudaGraphicsUnmapResources(1, &resources));
  CUDAERR(cudaGraphicsUnregisterResource(resources));
  CUDAERR(cudaFree(particles));
  CUDAERR(cudaFree(velocities));
}
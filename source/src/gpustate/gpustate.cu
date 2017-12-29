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

  const int block_size = 256;
  update_GL<<<(n + block_size - 1) / block_size, block_size>>>(particles, glptr, n);
}

void GPUState::clean() {
  CUDAERR(cudaGraphicsUnmapResources(1, &resources));
  CUDAERR(cudaGraphicsUnregisterResource(resources));
  CUDAERR(cudaFree(particles));
  CUDAERR(cudaFree(velocities));
}

__global__ void update_GL(CUParticle *particles, void *glptr, size_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  char *offset = (char*)(glptr) + 20*i;
  GLfloat *M_part = (GLfloat*)offset;
  GLuint *type_part = (GLuint*)(offset + 16);
  M_part[0] = particles[i].pos.x;
  M_part[1] = particles[i].pos.y;
  M_part[2] = particles[i].pos.z;
  M_part[3] = 1.0f;
  type_part[0] = particles[i].type;
}

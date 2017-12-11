#include "kernel.cuh"

__global__ void calculate_velocities(CUParticle *particles, float3 *velocities,
                                     size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  velocities[i] = make_float3(0.0, 0.0, 0.0);

  const double D = 376.78f;
  const double epsilon = 47.0975f;
  const double M[4] = {1.9549 * std::pow(10, 10), 7.4161 * std::pow(10, 9),
                       1.9549 * std::pow(10, 10), 7.4161 * std::pow(10, 9)};
  const double K[4] = {5.8228 * std::pow(10, 14), 2.29114 * std::pow(10, 13),
                       5.8228 * std::pow(10, 14), 2.29114 * std::pow(10, 13)};
  const double KRP[4] = {0.02, 0.01, 0.02, 0.01};
  const double SDP[4] = {0.002, 0.001, 0.002, 0.001};
  const double G = 6.67408;

  const auto p_i = particles[i];

  for (size_t j = 0; j < n; ++j) {
    if (j == i) {
      continue;
    }

    const auto p_j = particles[j];
    const auto diff = p_i.pos - p_j.pos;
    const float r = norm3df(diff.x, diff.y, diff.z);
    const auto dir = diff / r;
    double force;
  }
}

__global__ void apply_velocities(CUParticle *particles, float3 *velocities,
                                 size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  const auto vel = particles[i].velocity + velocities[i];
  particles[i].pos += vel;
}

void update(WorldState *world, float dt) {
  const auto N = world->particles.size();
  const auto block_size = 256;

  CUParticle *d_p = 0;
  float3 *d_v = 0;

  CUDAERR(cudaMalloc(&d_p, N * sizeof(*d_p)));
  CUDAERR(cudaMalloc(&d_v, N * sizeof(*d_v)));

  // glm is memory compatible with float3 so a struct should be aswell
  CUParticle *particles =
    reinterpret_cast<CUParticle *>(world->particles.data());
  CUDAERR(cudaMemcpy(d_p, particles, N * sizeof(*d_p), cudaMemcpyHostToDevice));

  calculate_velocities<<<(N + block_size - 1) / block_size, block_size>>>(
      d_p, d_v, N, dt);
  apply_velocities<<<(N + block_size - 1) / block_size, block_size>>>(d_p, d_v,
                                                                      N, dt);

  CUDAERR(cudaMemcpy(particles, d_p, N * sizeof(*particles),
                     cudaMemcpyDeviceToHost));

  CUDAERR(cudaFree(d_p));
  CUDAERR(cudaFree(d_v));
}

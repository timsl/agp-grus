#include "kernel.cuh"

__global__ void calculate_velocities(const CUParticle *particles, float3 *velocities,
                                     size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  velocities[i] = make_float3(0.0, 0.0, 0.0);

  const double D = 376.78f;
  const double D2 = std::pow(D, 2);
  const double epsilon = 47.0975f;
  const double M[4] = {1.9549 * std::pow(10, 10), 7.4161 * std::pow(10, 9),
                       1.9549 * std::pow(10, 10), 7.4161 * std::pow(10, 9)};
  const double K[4] = {5.8228 * std::pow(10, 14), 2.29114 * std::pow(10, 13),
                       5.8228 * std::pow(10, 14), 2.29114 * std::pow(10, 13)};
  const double KRP[4] = {0.02, 0.01, 0.02, 0.01};
  const double SDP[4] = {0.002, 0.001, 0.002, 0.001};
  const double G = 6.67408;

  const float3 p_i = particles[i].pos;
  const float3 v_i = particles[i].velocity;
  const char t_i = particles[i].type;

  float3 my_new_velocity = make_float3(0.0, 0.0, 0.0);

  for (size_t j = 0; j < n; ++j) {
    if (j == i) {
      continue;
    }

    const float3 p_j = particles[j].pos;
    const float3 v_j = particles[j].velocity;
    const char t_j = particles[j].type;

    const auto diff = p_j - p_i;
    const auto next_diff = ((p_j + v_j) - (p_i + v_i));

    double r = norm3df(diff.x, diff.y, diff.z);
    const double next_r = norm3df(next_diff.x, next_diff.y, next_diff.z);

    const auto dir = diff / r;
    double force = 0.0;

    // pre-computed values
    r = max(r, epsilon);
    const double r2 = std::pow(r, 2);
    const double gmm = G * M[t_i] * M[t_j] * std::pow(r, -2);
    const double dmr = (D2 - r2) * 0.5;

    if (r >= D) {
      // Not in contact
      force = gmm;
    } else if (r >= D - D * SDP[t_i]) {
      // In contact, but no shell penetrated
      force = gmm - dmr * (K[t_i] + K[t_j]);
    } else if (r >= D - D * SDP[t_j]){

    }

    my_new_velocity += dir * (float)(force * dt * std::pow(10, -14));
  }

  velocities[i] = my_new_velocity;
}

__global__ void apply_velocities(CUParticle *particles, float3 *velocities,
                                 size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  particles[i].velocity += velocities[i];
  particles[i].pos += particles[i].velocity;
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

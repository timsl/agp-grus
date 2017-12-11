#include "kernel.cuh"

__device__ float3 body_body_interaction(CUParticle pi, CUParticle pj) {
  const double D = 376.78;
  const double D2 = pow(D, 2);
  const double epsilon = 47.0975;
  const double M[4] = {1.9549 * pow(10, 10), 7.4161 * pow(10, 9),
                       1.9549 * pow(10, 10), 7.4161 * pow(10, 9)};
  const double K[4] = {5.8228 * pow(10, 14), 2.29114 * pow(10, 14),
                       5.8228 * pow(10, 14), 2.29114 * pow(10, 14)};
  const double KRP[4] = {0.02, 0.01, 0.02, 0.01};
  const double SDP[4] = {0.002, 0.001, 0.002, 0.001};
  const double G = 6.67408; // * 10^-20, but removed that from M and here

  // These scales are probably not necessary if we use better numerical techs
  const double weirdscale1 = pow(10, -16);
  const double weirdscale2 = pow(10, -22);

  const float3 p_i = pi.pos;
  const float3 v_i = pi.velocity;
  const char t_i = pi.type;

  const float3 p_j = pj.pos;
  const float3 v_j = pj.velocity;
  const char t_j = pj.type;

  const auto diff = p_j - p_i;
  const auto next_diff = ((p_j + v_j * 0.00001) - (p_i + v_i * 0.00001));

  double r = norm3d(diff.x, diff.y, diff.z);
  const double next_r = norm3d(next_diff.x, next_diff.y, next_diff.z);

  const auto dir = diff / r;
  double force = 0.0;

  // pre-computed values
  r = fmax(r, epsilon);
  const double r2 = pow(r, 2);
  const double gmm = G * M[t_i] * M[t_j] * pow(r, -2) * weirdscale1;
  const double dmr = (D2 - r2) * 0.5 * weirdscale2;
  const double oneshell = fmin(SDP[t_i], SDP[t_j]);
  const double twoshell = fmax(SDP[t_i], SDP[t_j]);

  if (r >= D) {
    // Not in contact
    force = gmm;
  } else if (r >= D - D * oneshell) {
    // In contact, but no shell penetrated
    force = gmm - dmr * (K[t_i] + K[t_j]);
  } else if (r >= D - D * twoshell) {
    // One shell has been penetrated
    if (next_r < r) {
      force = gmm - dmr * (K[t_i] + K[t_j]);
    } else {
      force = gmm - dmr * (K[t_i] * KRP[t_i] + K[t_j]);
    }
  } else {
    // Both shells penetrated (r > epsilon)
    if (next_r < r) {
      force = gmm - dmr * (K[t_i] + K[t_j]);
    } else {
      force = gmm - dmr * (K[t_i] * KRP[t_i] + K[t_j] * KRP[t_j]);
    }
  }

  return dir * (float)force;
}

__global__ void calculate_velocities(const CUParticle *particles,
                                     float3 *velocities, size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // Want a local velocity to write to in each loop to save some global
  // bandwidth
  float3 vel_acc = make_float3(0.0, 0.0, 0.0);

  for (size_t j = 0; j < n; ++j) {
    if (j == i) {
      continue;
    }

    vel_acc += body_body_interaction(particles[i], particles[j]);
  }

  velocities[i] = vel_acc * dt;
}

__global__ void apply_velocities(CUParticle *particles, float3 *velocities,
                                 size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  particles[i].velocity += velocities[i];
  particles[i].pos += particles[i].velocity * dt;
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

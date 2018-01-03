#include "kernel.cuh"

__device__ float3 body_body_interaction(CUParticle pi, CUParticle pj) {
  static const double D = 376.78;
  static const double epsilon = 47.0975;
  static const double M[4] = {1.9549e20, 7.4161e19,
                              1.9549e20, 7.4161e19};
  static const double K[4] = {5.8228e14, 2.29114e14,
                              5.8228e14, 2.29114e14};
  static const double KRP[4] = {0.02, 0.01, 0.02, 0.01};
  static const double SDP[4] = {0.002, 0.001, 0.002, 0.001};
  static const double G = 6.67408e-20;

  // These arbitrary scalings might not necessary if we use better
  // numerical techniques
  static const double weirdscale1 = 1e-16;
  static const double weirdscale2 = 1e-16;

  const auto diff = pj.pos - pi.pos;
  const auto next_diff =
    ((pj.pos + pj.velocity * 1e-5) - (pi.pos + pi.velocity * 1e-5));

  double r = length(diff);
  const double next_r = length(next_diff);

  const auto dir = diff / r;
  double force = 0.0;

  // pre-computed values
  r = fmax(r, epsilon);
  const double r2 = r*r;
  const double gmm = G * M[pi.type] * M[pj.type] * (1/r2) * weirdscale1;
  const double dmr = (D*D - r2) * 0.5 * weirdscale2;
  const double oneshell = fmin(SDP[pi.type], SDP[pj.type]);
  const double twoshell = fmax(SDP[pi.type], SDP[pj.type]);

  if (r >= D) {
    // Not in contact
    force = gmm;
  } else if (r >= D - D * oneshell) {
    // In contact, but no shell penetrated
    force = gmm - dmr * (K[pi.type] + K[pj.type]);
  } else if (r >= D - D * twoshell) {
    // One shell has been penetrated
    if (next_r < r) {
      force = gmm - dmr * (K[pi.type] + K[pj.type]);
    } else {
      force = gmm - dmr * (K[pi.type] * KRP[pi.type] + K[pj.type]);
    }
  } else {
    // Both shells penetrated (r > epsilon)
    if (next_r < r) {
      force = gmm - dmr * (K[pi.type] + K[pj.type]);
    } else {
      force =
        gmm - dmr * (K[pi.type] * KRP[pi.type] + K[pj.type] * KRP[pj.type]);
    }
  }

  return dir * (float)force;
}

__global__ void calculate_forces(const CUParticle *particles,
                                 float3 *forces, size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // Want a local force to write to in each loop to save some global
  // bandwidth
  const CUParticle my_part = particles[i];
  float3 force_acc = make_float3(0.0, 0.0, 0.0);

  extern __shared__ CUParticle sh_part[];

  const size_t sync_size = blockDim.x;
  const size_t sync_points = n / sync_size;

  for (size_t sync = 0; sync < sync_points; ++sync) {
    // read global memory and put in sh_part instead.
    // put in some j corresponding to this threads idx.
    sh_part[threadIdx.x] = particles[sync * sync_size + threadIdx.x];
    __syncthreads();

    for (size_t j = 0; j < sync_size; ++j) {
      if (sync * sync_size + j == i) {
        continue;
      }

      force_acc += body_body_interaction(my_part, sh_part[j]);
    }
    __syncthreads();
  }

  forces[i] = force_acc;
}

__global__ void apply_forces(CUParticle *particles, float3 *forces,
                             size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // We always update pos just after velocity to avoid another kernel.
  particles[i].velocity += forces[i] * dt;
  particles[i].pos += particles[i].velocity * dt;
}

__global__ void first_apply_forces(CUParticle *particles, float3 *forces,
                                   size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // First leapfrog step v_{1/2}, need to update vel by only dt/2.
  // We always update pos just after velocity to avoid another kernel.
  particles[i].velocity += forces[i] * dt/2;
  particles[i].pos += particles[i].velocity * dt;
}

void first_update(WorldState *world, float dt) {
  const auto N = world->particles.size();
  const auto block_size = 32;

  calculate_forces<<<(N + block_size - 1) / block_size, block_size,
                         block_size * sizeof(CUParticle)>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  first_apply_forces<<<(N + block_size - 1) / block_size, block_size>>>(
                                                                        world->gpu.particles, world->gpu.velocities, N, dt);

  update_GL<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.glptr, N);
}

void update(WorldState *world, float dt) {
  const auto N = world->particles.size();
  const auto block_size = 32;

  calculate_forces<<<(N + block_size - 1) / block_size, block_size,
                         block_size * sizeof(CUParticle)>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  apply_forces<<<(N + block_size - 1) / block_size, block_size>>>(
                                                                  world->gpu.particles, world->gpu.velocities, N, dt);

  update_GL<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.glptr, N);

  // CUDAERR(cudaDeviceSynchronize());

  // CUParticle *cast = reinterpret_cast<CUParticle *>(world->particles.data());
  // CUDAERR(cudaMemcpy(cast, world->gpu.particles, N * sizeof(*cast),
  //                    cudaMemcpyDeviceToHost));
}

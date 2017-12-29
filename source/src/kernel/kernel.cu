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

  const auto diff = pj.pos - pi.pos;
  const auto next_diff =
      ((pj.pos + pj.velocity * 0.00001) - (pi.pos + pi.velocity * 0.00001));

  double r = norm3d(diff.x, diff.y, diff.z);
  const double next_r = norm3d(next_diff.x, next_diff.y, next_diff.z);

  const auto dir = diff / r;
  double force = 0.0;

  // pre-computed values
  r = fmax(r, epsilon);
  const double r2 = pow(r, 2);
  const double gmm = G * M[pi.type] * M[pj.type] * pow(r, -2) * weirdscale1;
  const double dmr = (D2 - r2) * 0.5 * weirdscale2;
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

__global__ void calculate_velocities(const CUParticle *particles,
                                     float3 *velocities, size_t n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // Want a local velocity to write to in each loop to save some global
  // bandwidth
  const CUParticle my_part = particles[i];
  float3 vel_acc = make_float3(0.0, 0.0, 0.0);

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

      vel_acc += body_body_interaction(my_part, sh_part[j]);
    }
    __syncthreads();
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
  const auto block_size = 32;

  calculate_velocities<<<(N + block_size - 1) / block_size, block_size,
                         block_size * sizeof(CUParticle)>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  apply_velocities<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);

  update_GL<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.glptr, N);

  // CUDAERR(cudaDeviceSynchronize());

  // CUParticle *cast = reinterpret_cast<CUParticle *>(world->particles.data());
  // CUDAERR(cudaMemcpy(cast, world->gpu.particles, N * sizeof(*cast),
  //                    cudaMemcpyDeviceToHost));
}

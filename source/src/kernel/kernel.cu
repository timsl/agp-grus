#include "kernel.cuh"

__device__ float3 body_body_interaction(const CUParticle pi,
                                        const CUParticle pj) {
  static const float D = 420;
  static const float epsilon = 47.0975;
  static const float M[4] = {1.9549e20, 7.4161e19};
  static const float K[4] = {5.8228e14, 2.29114e14};
  static const float KRP[4] = {0.02, 0.01};
  static const float SDP[4] = {0.002, 0.001};
  static const float G = 6.67408e-20;

  // These arbitrary scalings might not necessary if we use better
  // numerical techniques
  static const float weirdscale1 = 1e-16;
  static const float weirdscale2 = 1e-16;

  const auto diff = pj.pos - pi.pos;
  const auto next_diff =
      ((pj.pos + pj.velocity * 1e-5) - (pi.pos + pi.velocity * 1e-5));

  const int ti = pi.type >= 2 ? pi.type - 2 : pi.type;
  const int tj = pj.type >= 2 ? pj.type - 2 : pj.type;

  // Iron has a larger shell, and would get penetrated first in the
  // ifs.  Largest iron=0, smallest silicate=1.  We can use these
  // instead of ti, tj. If they're different then these will be the
  // two different ones, if they're the same they will be equal to
  // ti=tj.
  const auto tlarge = min(ti, tj);
  const auto tsmall = max(ti, tj);

  const float r = fmax(length(diff), epsilon);
  const float next_r = length(next_diff);

  const auto dir = diff / r;

  // pre-computed values
  const float r2 = r * r;
  const float gmm = G * M[ti] * M[tj] * (1 / r2) * weirdscale1;
  const float dmr = (D * D - r2) * 0.5 * weirdscale2;

  float force = gmm;
  if (r < D) {
    float KRPlarge =
        next_r > r && r <= D * (1.0 - SDP[tlarge]) ? KRP[tlarge] : 1.0;
    float KRPsmall =
        next_r > r && r <= D * (1.0 - SDP[tsmall]) ? KRP[tsmall] : 1.0;
    force -= dmr * (K[tsmall] * KRPsmall + K[tlarge] * KRPlarge);
  }
  return dir * force;
}

__global__ void calculate_forces(const CUParticle *particles, float3 *forces,
                                 size_t n, float dt) {
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

__global__ void apply_forces(CUParticle *particles, float3 *forces, size_t n,
                             float dt) {
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
  particles[i].velocity += forces[i] * dt / 2;
  particles[i].pos += particles[i].velocity * dt;
}

void first_update(WorldState *world, float dt) {
  const auto N = world->particles.size();
  const auto block_size = world->block_size;

  calculate_forces<<<(N + block_size - 1) / block_size, block_size,
                     block_size * sizeof(CUParticle)>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  CUDAERR(cudaPeekAtLastError());
  first_apply_forces<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  CUDAERR(cudaPeekAtLastError());

  update_GL<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.glptr, N);
  CUDAERR(cudaPeekAtLastError());

  // Synchronize CUDA so that the timings are correct
  CUDAERR(cudaDeviceSynchronize());
}

void update(WorldState *world, float dt) {
  const auto N = world->particles.size();
  const auto block_size = world->block_size;

  calculate_forces<<<(N + block_size - 1) / block_size, block_size,
                     block_size * sizeof(CUParticle)>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  CUDAERR(cudaPeekAtLastError());
  apply_forces<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.velocities, N, dt);
  CUDAERR(cudaPeekAtLastError());

  update_GL<<<(N + block_size - 1) / block_size, block_size>>>(
      world->gpu.particles, world->gpu.glptr, N);
  CUDAERR(cudaPeekAtLastError());

  // Synchronize CUDA so that the timings are correct
  CUDAERR(cudaDeviceSynchronize());
}

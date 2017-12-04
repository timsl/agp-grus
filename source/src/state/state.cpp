#include "state.hpp"

void WorldState::update(float dt, float t) {
  auto n = particles.size();

  for (unsigned i = 0; i < n; ++i) {
    auto &p = particles[i];

    float inperiod = (float)i / (float)n + t / PERIOD;
    inperiod *= 2 * M_PI;

    p.velocity.x = cos(inperiod) / 50;
    p.velocity.y = sin(inperiod) / 50;
    p.velocity.z = 0.0f; // 1.0f / 800;
    p.pos += p.velocity * dt;
  }
}

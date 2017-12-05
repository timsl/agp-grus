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

void WorldState::create_planet(std::vector<Particle> &particles, size_t nr_inner, size_t nr_outer, float radius_1, float radius_2, glm::vec3 planet_origin) {
    // used to generate our random numbers
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (size_t i = 0; i < nr_inner; ++i) {
        float rho_1 = dist(generator);
        float rho_2 = dist(generator);
        float rho_3 = dist(generator);

        float Rp_1 = radius_1 * std::pow(rho_1, (1.0f / 3.0f));
        float u = 1.0f - 2.0f * rho_2;
        float sqrt_u2 = std::sqrt(1 - u * u);

        particles[i].pos.x = planet_origin.x + Rp_1 * sqrt_u2 * std::cos(2 * M_PI * rho_3);
        particles[i].pos.y = planet_origin.y + Rp_1 * sqrt_u2 * std::sin(2 * M_PI * rho_3);
        particles[i].pos.z = planet_origin.z + Rp_1 * u;
        particles[i].type = 0;
    }
   
    // silicate
    for (size_t i = nr_inner; i < nr_inner + nr_outer; ++i) {
        float rho_1 = dist(generator);
        float rho_2 = dist(generator);
        float rho_3 = dist(generator);

        float R = std::pow(radius_1, 3) + (std::pow(radius_2, 3) - std::pow(radius_1, 3)) * rho_1;
        float Rp_1 = std::pow(R, 1.0f/3.0f);
        float u = 1.0f - 2.0f * rho_2;
        float sqrt_u2 = std::sqrt(1 - u * u);

        particles[i].pos.x = planet_origin.x + Rp_1 * sqrt_u2 * std::cos(2 * M_PI * rho_3);
        particles[i].pos.y = planet_origin.y + Rp_1 * sqrt_u2 * std::sin(2 * M_PI * rho_3);
        particles[i].pos.z = planet_origin.z + Rp_1 * u;
    }
}


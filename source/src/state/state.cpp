#include "state.hpp"

void WorldState::update(float dt) {
  auto n = particles.size();
  const double D = 376.78f;
  const double epsilon = 47.0975f;
  const double M[4] = {1.9549 * std::pow(10, 10), 7.4161 * std::pow(10, 9),
                       1.9549 * std::pow(10, 10), 7.4161 * std::pow(10, 9)};
  const double K[4] = {5.8228 * std::pow(10, 14), 2.29114 * std::pow(10, 13),
                       5.8228 * std::pow(10, 14), 2.29114 * std::pow(10, 13)};
  const double KRP[4] = {0.02, 0.01, 0.02, 0.01};
  const double SDP[4] = {0.002, 0.001, 0.002, 0.001};
  const double G = 6.67408;

  for (size_t i = 0; i < n; ++i) {
    auto &p_i = particles[i];

    for (size_t j = 0; j < n; ++j) {
      if (j != i) {
        auto &p_j = particles[j];
        float r = glm::distance(p_i.pos, p_j.pos);
        auto dir = glm::normalize(p_j.pos - p_i.pos);
        double force;

        int pi_t = (int)p_i.type;
        int pj_t = (int)p_j.type;

        // If r is too small set it to epsilon
        if (r < epsilon) {
          r = epsilon;
        }

        // Elements are not in contact
        if (D <= r) {
          force = G * M[pi_t] * M[pj_t] * std::pow(r, -2);
        }
        // Element are in contact but no shell is penetrated
        else if (D - D * SDP[pi_t] <= r && r < D) {
          force =
              G * M[pi_t] * M[pj_t] * std::pow(r, -2) -
              0.5f * (K[pi_t] + K[pj_t]) * (std::pow(D, 2) - std::pow(r, 2));
        }
        // One shell has been penetrated
        else if (D - D * SDP[pj_t] <= r && r < D - D * SDP[pi_t]) {
          float next_r =
              glm::distance(p_i.pos + p_i.velocity, p_j.pos + p_j.velocity);
          // Seperation is decreasing
          if (next_r < r) {
            force =
                G * M[pi_t] * M[pj_t] * std::pow(r, -2) -
                0.5f * (K[pi_t] + K[pj_t]) * (std::pow(D, 2) - std::pow(r, 2));

          }
          // Seperation is increasing
          else {
            force = G * M[pi_t] * M[pj_t] * std::pow(r, -2) -
                     0.5f * (K[pi_t] * KRP[pi_t] + K[pj_t]) *
                         (std::pow(D, 2) - std::pow(r, 2));
          }
        }
        // Both shells have been penetrated
        else if (epsilon <= r && r < D - D * SDP[pj_t]) {
          float next_r =
              glm::distance(p_i.pos + p_i.velocity, p_j.pos + p_j.velocity);
          // Seperation is decreasing
          if (next_r < r) {
            force =
                G * M[pi_t] * M[pj_t] * std::pow(r, -2) -
                0.5f * (K[pi_t] + K[pj_t]) * (std::pow(D, 2) - std::pow(r, 2));

          }
          // Seperation is increasing
          else {
            force = G * M[pi_t] * M[pj_t] * std::pow(r, -2) -
                     0.5f * (K[pi_t] * KRP[pi_t] + K[pj_t] * KRP[pj_t]) *
                         (std::pow(D, 2) - std::pow(r, 2));
          }
        }
        p_i.velocity += dir * (float)(force * dt * std::pow(10, -14));
      }
    }
    p_i.pos += p_i.velocity * dt;
  }
}

void WorldState::create_planets(std::vector<Particle> &particles,
                                float radius_1, float radius_2,
                                float procent_iron, glm::vec3 planet_1_origin,
                                glm::vec3 planet_2_origin) {
  int half = particles.size() / 2;
  int nr_iron = half * procent_iron;
  int nr_silicate = half - nr_iron;

  auto inital_velocity = glm::vec3(-3.2416f, 0.0f, 0.0f); // km/s
  float omega = 3.0973f / 360.0f;                         // rad s⁻1

  auto start = particles.begin();
  create_sphere(start, start + nr_iron, 0, radius_1, planet_1_origin,
                inital_velocity, iron_1, omega);

  start += nr_iron;
  create_sphere(start, start + nr_silicate, radius_1, radius_2, planet_1_origin,
                inital_velocity, silicate_1, omega);

  start += nr_silicate;
  create_sphere(start, start + nr_iron, 0, radius_1, planet_2_origin,
                -1.0f * inital_velocity, iron_2, -1.0f * omega);

  start += nr_iron;
  create_sphere(start, start + nr_silicate, radius_1, radius_2, planet_2_origin,
                -1.0f * inital_velocity, silicate_2, -1.0f * omega);
}

template <typename Iter>
void WorldState::create_sphere(Iter start, Iter end, float radius_1,
                               float radius_2, glm::vec3 planet_origin,
                               glm::vec3 inital_velocity, char prop_type,
                               float omega) {
  // used to generate our random numbers
  std::default_random_engine generator;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  // silicate
  for (auto i = start; i != end; ++i) {
    // positon
    float rho_1 = dist(generator);
    float rho_2 = dist(generator);
    float rho_3 = dist(generator);

    float R = std::pow(radius_1, 3) +
              (std::pow(radius_2, 3) - std::pow(radius_1, 3)) * rho_1;
    float Rp_1 = std::pow(R, 1.0f / 3.0f);
    float u = 1.0f - 2.0f * rho_2;
    float sqrt_u2 = std::sqrt(1 - u * u);

    i->type = prop_type;
    
    i->pos.x = planet_origin.x + Rp_1 * sqrt_u2 * std::cos(2 * M_PI * rho_3);
    i->pos.y = planet_origin.y + Rp_1 * sqrt_u2 * std::sin(2 * M_PI * rho_3);
    i->pos.z = planet_origin.z + Rp_1 * u;

    // velocity
    i->velocity = inital_velocity;
    float r_xz = std::sqrt(std::pow(i->pos.x - planet_origin.x, 2) +
                           std::pow(i->pos.z - planet_origin.z, 2));
    float theta =
        std::atan((i->pos.z - planet_origin.z) / (i->pos.x - planet_origin.x));

    auto added_velocity = glm::vec3(omega * r_xz * std::sin(theta), 0.0f,
                                    -1.0f * omega * r_xz * std::cos(theta));
    i->velocity += added_velocity;
  }
}

void CameraState::move(glm::vec3 translation) { pos += translation; }
void CameraState::rotate(glm::vec3 axis, float angle) {
  glm::mat4 R = glm::rotate(angle, axis);
  auto new_dir = glm::vec3(R * glm::vec4(dir, 1.0f));
  auto new_up = glm::vec3(R * glm::vec4(up, 1.0f));
  dir = glm::normalize(new_dir);
  up = glm::normalize(new_up);
}

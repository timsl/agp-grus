#include "state.hpp"

void WorldState::update(float dt, float t) {
  auto n = particles.size();
  const float D = 376.78;
  const float epsilon = 47.0975;
  const double M[4] = {1.9549 * std::pow(10, 20), 7.4161 * std::pow(10, 19),
                       1.9549 * std::pow(10, 20), 7.4161 * std::pow(10, 19)};
  const double K[4] = {2.9114 * std::pow(10, 11), 7.2785 * std::pow(10, 10),
                       2.9114 * std::pow(10, 11), 7.2785 * std::pow(10, 10)};
  const float KRP[4] = {0.02, 0.01, 0.02, 0.01};
  const float SDP[4] = {0.01, 0.001, 0.01, 0.001};
  const double G = 6.67408 * std::pow(10, -11);

  for (size_t i = 0; i < n; ++i) {
    auto &p_i = particles[i];

    for (size_t j = 0; j < n; ++j) {
      if (j != i) {
        auto &p_j = particles[j];
        float r = glm::distance(p_i.pos, p_j.pos);
        auto force = glm::normalize(p_j.pos - p_i.pos);

        if (r < epsilon) {
            r = epsilon;
        }

        if (D <= r) {
            force *= G * M[(int)p_i.type] * M[(int)p_j.type] * std::pow(r, -2);
        } else if (D - D * SDP[(int) p_j.type] <= r && r < D) {
            force *= G * M[(int)p_i.type] * M[(int)p_j.type] * std::pow(r, -2) - 0.5f * (K[(int)p_i.type] + K[(int)p_j.type]) * (std::pow(D, 2) - std::pow(r, 2));

        }
        p_i.velocity += force;
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

  // auto inital_velocity = glm::vec3(3.2416f, 0.0f, 0.0f); // km/s
  auto inital_velocity = glm::vec3(0.032416f, 0.0f, 0.0f); // km/s
  float omega = 0.00844638888f;                            // rad s⁻1

  auto start = particles.begin();
  create_sphere(start, start + nr_iron, 0, radius_1, planet_1_origin,
                inital_velocity, iron_1, omega);

  start += nr_iron;
  create_sphere(start, start + nr_silicate, radius_1, radius_2, planet_1_origin,
                inital_velocity, silicate_1, omega);

  start += nr_silicate;
  create_sphere(start, start + nr_iron, 0, radius_1, planet_2_origin,
                -1.0f * inital_velocity, iron_2, omega);

  start += nr_iron;
  create_sphere(start, start + nr_silicate, radius_1, radius_2, planet_2_origin,
                -1.0f * inital_velocity, silicate_2, omega);
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

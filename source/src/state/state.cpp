#include "state.hpp"

void WorldState::create_planets(std::vector<Particle> &particles,
                                float radius_1, float radius_2,
                                float ratio_iron, glm::vec3 planet_1_origin,
                                glm::vec3 planet_2_origin, bool use_rotation) {
  int half = particles.size() / 2;
  int nr_iron = half * ratio_iron;
  int nr_silicate = half - nr_iron;

  auto initial_velocity = glm::vec3(-3.2416f, 0.0f, 0.0f); // km/s
  float omega = 3.0973 / 3600.0;                           // rad s⁻1

  auto start = particles.begin();
  create_sphere(start, start + nr_iron, 0, radius_1, planet_1_origin,
                initial_velocity, iron_1, omega, use_rotation);

  start += nr_iron;
  create_sphere(start, start + nr_silicate, radius_1, radius_2, planet_1_origin,
                initial_velocity, silicate_1, omega, use_rotation);

  start += nr_silicate;
  create_sphere(start, start + nr_iron, 0, radius_1, planet_2_origin,
                -1.0f * initial_velocity, iron_2, -1.0f * omega, use_rotation);

  start += nr_iron;
  create_sphere(start, start + nr_silicate, radius_1, radius_2, planet_2_origin,
                -1.0f * initial_velocity, silicate_2, -1.0f * omega,
                use_rotation);
}

template <typename Iter>
void WorldState::create_sphere(Iter start, Iter end, float radius_1,
                               float radius_2, glm::vec3 planet_origin,
                               glm::vec3 initial_velocity, char prop_type,
                               float omega, bool use_rotation) {
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
    i->velocity = initial_velocity;
    float r_xz = std::sqrt(std::pow(i->pos.x - planet_origin.x, 2) +
                           std::pow(i->pos.z - planet_origin.z, 2));
    float theta =
        std::atan2((i->pos.z - planet_origin.z), (i->pos.x - planet_origin.x));

    auto added_velocity = glm::vec3(omega * r_xz * std::sin(theta), 0.0f,
                                    -1.0f * omega * r_xz * std::cos(theta));
    if (use_rotation) {
      i->velocity += added_velocity;
    }
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

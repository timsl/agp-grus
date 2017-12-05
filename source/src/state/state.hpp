#ifndef STATE_HPP
#define STATE_HPP

#include <random>
#include <math.h>
#include "common.hpp"

const int PERIOD = 100;
enum PlanetProperty : char {iron_1, silicate_1, iron_2, silicate_2};

struct Particle {
  glm::vec3 pos;
  glm::vec3 velocity;
  char type;
};

struct CameraState {
  float angle;
  float fov;

  CameraState(float angle = 0.0f, float fov = 90.0f) : angle(angle), fov(fov) {}
};

struct WorldState {
  CameraState cam;
  std::vector<Particle> particles;

  void update(float dt, float t);
  void create_planet(std::vector<Particle> &particles, size_t nr_inner, size_t nr_outer, float radius_1, float radius_2, glm::vec3 planet_origin);
  WorldState(int n) : particles(n) {}
};

#endif

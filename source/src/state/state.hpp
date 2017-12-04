#ifndef STATE_HPP
#define STATE_HPP

#include "common.hpp"

const int PERIOD = 100;

struct Particle {
  glm::vec3 pos;
  glm::vec3 velocity;
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
  WorldState(int n) : particles(n) {}
};

#endif

#ifndef STATE_HPP
#define STATE_HPP

#include "common.hpp"
#include "heldactions.hpp"
#include <math.h>
#include <random>

const int PERIOD = 100;
enum PlanetProperty : char {
  iron_1 = 0,
  silicate_1 = 1,
  iron_2 = 2,
  silicate_2 = 3
};

struct ParticleProps {
  float inner_radius;
  float outer_radius;
  glm::vec4 color;
};

struct Particle {
  glm::vec3 pos;
  glm::vec3 velocity;
  char type;
};

struct CameraState {
  float fov = 60.0f;
  double old_xpos = 0, old_ypos = 0;
  glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 dir = glm::vec3(0.0f, 0.0f, 1.0f);
  glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

  CameraState() {}
  void move(glm::vec3 dir);
  void rotate(glm::vec3 axis, float angle);
};

struct WindowState {
  int width;
  int height;

  WindowState() : width(1280), height(720) {}
};

struct WorldState {
  CameraState cam;
  std::vector<Particle> particles;
  WindowState window;
  HeldActions held;
  std::vector<ParticleProps> particle_props;

  void update(float dt, float t);

  void create_planets(std::vector<Particle> &particles, float radius_1,
                      float radius_2, float procent_iron,
                      glm::vec3 planet_1_origin, glm::vec3 planet_2_origin);
  template <typename Iter>
  void create_sphere(Iter start, Iter end, float radius_1, float radius_2,
                     glm::vec3 planet_origin, glm::vec3 inital_velocity,
                     char prop_type, float omega);
  WorldState(int n) : particles(n), particle_props(4) {}
};

#endif

#ifndef STATE_HPP
#define STATE_HPP

#include "common.hpp"
#include "gpustate.cuh"
#include "heldactions.hpp"
#include "sphere.hpp"
#include <math.h>
#include <random>

const int PERIOD = 100;
enum PlanetProperty : char {
  iron_1 = 0,
  silicate_1 = 1,
  iron_2 = 2,
  silicate_2 = 3
};

struct Particle {
  glm::vec3 pos;
  glm::vec3 velocity;
  char type;
};

struct CameraState {
  float fov = 60.0f;
  double old_xpos = 0, old_ypos = 0;
  bool has_seen_mouse = false;
  glm::vec3 pos = glm::vec3(0.0f, 40000.0f, 0.0f);
  glm::vec3 dir = glm::vec3(0.0f, -1.0f, 0.0f);
  glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);

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
  WindowState window;
  HeldActions held;
  GPUState gpu;

  std::vector<Particle> particles;
  std::vector<glm::vec4> colors;

  Sphere *sphere;

  void create_planets(std::vector<Particle> &particles, float radius_1,
                      float radius_2, float procent_iron,
                      glm::vec3 planet_1_origin, glm::vec3 planet_2_origin,
                      bool use_rotation);
  template <typename Iter>
  void create_sphere(Iter start, Iter end, float radius_1, float radius_2,
                     glm::vec3 planet_origin, glm::vec3 inital_velocity,
                     char prop_type, float omegau, bool use_rotation);
  WorldState(int n) : particles(n), colors(4) {
    colors[0] = glm::vec4(0.83137f, 0.25098f, 0.14510f, 0.4f);
    colors[1] = glm::vec4(0.03922f, 0.20784f, 0.21176f, 0.4f);
    colors[2] = glm::vec4(0.84706f, 0.70588f, 0.36471f, 0.4f);
    colors[3] = glm::vec4(0.11765f, 0.40392f, 0.35294f, 0.4f);
  }
};

#endif

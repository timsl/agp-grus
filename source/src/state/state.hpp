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

// Struct representing a particle.
struct Particle {
  glm::vec3 pos;
  glm::vec3 velocity;
  char type;
};

// Struct keeping the direction and location of the camera.
struct CameraState {
  float fov = 60.0f;
  double old_xpos = 0, old_ypos = 0;
  bool has_seen_mouse = false;
  glm::vec3 pos = glm::vec3(0.0f, 40000.0f, 0.0f);
  glm::vec3 dir = glm::vec3(0.0f, -1.0f, 0.0f);
  glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);

  CameraState() {}

  // Move the camera in direction.
  void move(glm::vec3 dir);

  // Rotate the camera around axis by angle.
  // Normalizes the camera afterwards.
  void rotate(glm::vec3 axis, float angle);
};

// Struct representing the window state.
struct WindowState {
  int width;
  int height;

  WindowState() : width(1280), height(720) {}
};

// Struct for "the world", as opposed to keeping stuff as global
// variables. Maintains pretty much everything about the program.
struct WorldState {
  CameraState cam;
  WindowState window;
  HeldActions held;
  GPUState gpu;

  std::vector<Particle> particles;
  std::vector<glm::vec4> colors;
  size_t block_size;

  Sphere *sphere;

  void create_planets(std::vector<Particle> &particles, float radius_1,
                      float radius_2, float procent_iron,
                      glm::vec3 planet_1_origin, glm::vec3 planet_2_origin,
                      bool use_rotation);
  template <typename Iter>
  void create_sphere(Iter start, Iter end, float radius_1, float radius_2,
                     glm::vec3 planet_origin, glm::vec3 inital_velocity,
                     char prop_type, float omegau, bool use_rotation);

  // Constructs everything, the un-mentioned structs get default
  // constructed.
  WorldState(int n, int block_size) : particles(n), colors(4), block_size(block_size) {
    colors[0] = glm::vec4(0.83137f, 0.25098f, 0.14510f, 0.4f);
    colors[1] = glm::vec4(0.03922f, 0.20784f, 0.21176f, 0.4f);
    colors[2] = glm::vec4(0.84706f, 0.70588f, 0.36471f, 0.4f);
    colors[3] = glm::vec4(0.11765f, 0.40392f, 0.35294f, 0.4f);
  }
};

#endif

#include "input.hpp"

void resize_callback(GLFWwindow *win, int width, int height,
                     WorldState *world) {
  world->window.width = width;
  world->window.height = height;
  glViewport(0, 0, width, height);
}

// Handle keypresses. Generally only one action per keypress, so we can return
// if anything matches.
void keyboard_callback(GLFWwindow *win, int key, int action,
                       WorldState *world) {

  // Keys that act when held down. REPEAT is considered the same as
  // press since the held thingies should be idempotent anyways.
  bool key_down = action == GLFW_PRESS || action == GLFW_REPEAT;
  switch (key) {
  case GLFW_KEY_LEFT:
    world->held.turn_left = key_down;
    return;
  case GLFW_KEY_RIGHT:
    world->held.turn_right = key_down;
    return;
  case GLFW_KEY_UP:
    world->held.turn_up = key_down;
    return;
  case GLFW_KEY_DOWN:
    world->held.turn_down = key_down;
    return;
  case GLFW_KEY_W:
    world->held.move_forw = key_down;
    return;
  case GLFW_KEY_A:
    world->held.move_left = key_down;
    return;
  case GLFW_KEY_S:
    world->held.move_back = key_down;
    return;
  case GLFW_KEY_D:
    world->held.move_right = key_down;
    return;
  case GLFW_KEY_Q:
    world->held.roll_left = key_down;
    return;
  case GLFW_KEY_E:
    world->held.roll_right = key_down;
    return;
  case GLFW_KEY_LEFT_SHIFT:
    world->held.move_up = key_down;
    return;
  case GLFW_KEY_LEFT_CONTROL:
    world->held.move_down = key_down;
    return;
  }

  // Keys that aren't held, but just act
  {
    bool p = action == GLFW_PRESS;
    switch (key) {
    case GLFW_KEY_ESCAPE:
      if (p) {
        glfwSetWindowShouldClose(win, 1);
      }
      return;
    case GLFW_KEY_R:
      if (p) {
        world->cam = CameraState();
      }
      return;
    case GLFW_KEY_P:
      if (p) {
        world->held.simulation_running ^= true;
      }
      return;
    case GLFW_KEY_PERIOD:
      if (p) {
        update(world, 0.1);
      }
      return;
    }
  }

  // Fallback when nothing matches
  const char *actionname = "ERROR_WEIRD_ACTION";
  if (action == GLFW_PRESS) {
    actionname = "Pressed";
  } else if (action == GLFW_REPEAT) {
    actionname = "Repeated";
  } else if (action == GLFW_RELEASE) {
    actionname = "Released";
  }
  printf("%s unbound key %d.\n", actionname, key);
}

constexpr double sensitivity = 0.1 * (M_PI / 180.0);
void cursor_callback(GLFWwindow *win, double xpos, double ypos,
                     WorldState *world) {
  auto &c = world->cam;
  if (!c.has_seen_mouse) {
    c.old_xpos = xpos;
    c.old_ypos = ypos;
    c.has_seen_mouse = true;
  }

  auto x_diff = xpos - c.old_xpos;
  c.old_xpos = xpos;
  auto y_diff = ypos - c.old_ypos;
  c.old_ypos = ypos;

  c.rotate(glm::cross(c.up, c.dir), y_diff * sensitivity);
  c.rotate(c.up, -x_diff * sensitivity);
}

constexpr float movespeed = 5000.0;
constexpr float rotationspeed = 15 * 3.6 * (M_PI / 180.0);
void update_held(WorldState *world, float dt) {
  const float ms = movespeed * dt;
  const float rs = rotationspeed * dt;
  auto &h = world->held;
  auto &c = world->cam;

  if (h.move_forw) {
    c.move(c.dir * ms);
  }
  if (h.move_back) {
    c.move(-c.dir * ms);
  }
  if (h.move_up) {
    c.move(c.up * ms);
  }
  if (h.move_down) {
    c.move(-c.up * ms);
  }
  if (h.move_left) {
    // These are normalized as the inputs are orthogonal
    c.move(glm::cross(c.up, c.dir) * ms);
  }
  if (h.move_right) {
    c.move(glm::cross(c.dir, c.up) * ms);
  }

  if (h.turn_left) {
    c.rotate(c.up, rs);
  }
  if (h.turn_right) {
    c.rotate(c.up, -rs);
  }
  if (h.turn_up) {
    c.rotate(glm::cross(c.dir, c.up), rs);
  }
  if (h.turn_down) {
    c.rotate(glm::cross(c.up, c.dir), rs);
  }
  if (h.roll_left) {
    c.rotate(c.dir, -rs);
  }
  if (h.roll_right) {
    c.rotate(c.dir, rs);
  }
}

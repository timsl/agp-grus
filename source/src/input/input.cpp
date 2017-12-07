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
  bool key_down = action != GLFW_RELEASE;
  switch (key) {
  case GLFW_KEY_LEFT:
    world->held.turn_left = key_down;
    return;
  case GLFW_KEY_RIGHT:
    world->held.turn_right = key_down;
    return;
  case GLFW_KEY_UP:
    world->held.turn_right = key_down;
    return;
  case GLFW_KEY_DOWN:
    world->held.turn_right = key_down;
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
    world->held.move_down = key_down;
    return;
  case GLFW_KEY_C:
    world->held.move_up = key_down;
    return;
  }

  // Not handling release just means the code below is a bit neater,
  // since I'm not using it for any of the current buttons
  if (action == GLFW_PRESS) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(win, 1);
      return;
    case GLFW_KEY_R:
      world->cam.angle = 0.0f;
      world->cam.fov = 90.0f;
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

void update_held(WorldState *w, float dt) {
  auto &h = w->held;
  if (h.turn_left) {
    w->cam.angle -= 5.0f * dt;
  }
  if (h.turn_right) {
    w->cam.angle += 5.0f * dt;
  }
}

#include "input.hpp"

void resize_callback(GLFWwindow *win, int width, int height, WorldState *world) {
  world->window.width = width;
  world->window.height = height;
  glViewport(0, 0, width, height);
}

void keyboard_callback(GLFWwindow *win, int key, int action, WorldState *world) {
  // Not handling release just means the code below is a bit neater,
  // since I'm not using it for any of the current buttons
  if (action == GLFW_RELEASE)
    return;

  switch (key) {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(win, 1);
    break;
  case GLFW_KEY_R:
    world->cam.angle = 0.0f;
    world->cam.fov = 90.0f;
    break;
  case GLFW_KEY_LEFT:
    world->cam.angle -= 5.0f;
    break;
  case GLFW_KEY_RIGHT:
    world->cam.angle += 5.0f;
    break;
  case GLFW_KEY_SLASH: // Where i expect + to be on swedish keyboard
    world->cam.fov += 2.0f;
    break;
  case GLFW_KEY_MINUS:
    world->cam.fov -= 2.0f;
    break;
  default:
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
}

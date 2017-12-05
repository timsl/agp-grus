#ifndef INPUT_HPP
#define INPUT_HPP

#include "common.hpp"
#include "state.hpp"

void resize_callback(GLFWwindow *win, int width, int height, WorldState *world);

void keyboard_callback(GLFWwindow *win, int key, int action, WorldState *world);

#endif

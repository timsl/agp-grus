#ifndef INPUT_HPP
#define INPUT_HPP

#include "common.hpp"
#include "heldactions.hpp"
#include "state.hpp"

void resize_callback(GLFWwindow *win, int width, int height, WorldState *world);
void cursor_callback(GLFWwindow *win, double xpos, double ypos,
                     WorldState *world);
void keyboard_callback(GLFWwindow *win, int key, int action, WorldState *world);

void update_held(WorldState *w, float dt);

#endif

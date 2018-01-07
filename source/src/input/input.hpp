#ifndef INPUT_HPP
#define INPUT_HPP

#include "common.hpp"
#include "heldactions.hpp"
#include "kernel.cuh"
#include "state.hpp"

// Handle resizing of window.
void resize_callback(GLFWwindow *win, int width, int height, WorldState *world);

// Handle movement of cursor.
void cursor_callback(GLFWwindow *win, double xpos, double ypos,
                     WorldState *world);

// Handle keypresses. Commonly just changes a state in the heldstruct,
// and relies on other update steps to act.
void keyboard_callback(GLFWwindow *win, int key, int action, WorldState *world);

// Act on whichever keys are held, moving/tilting etcetera.
void update_held(WorldState *w, float dt);

#endif

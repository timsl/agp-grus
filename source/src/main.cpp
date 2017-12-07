
#include "common.hpp"
#include "input.hpp"
#include "sphere.hpp"
#include "state.hpp"
#include "util.hpp"

using namespace std;
using namespace agp;

GLuint g_default_vao = 0;
GLint color_loc = -1;
GLint MVP_loc = -1;
GLuint shader_program;

constexpr const char *FRAG_FILE = "src/shaders/frag.glsl";
constexpr const char *VERT_FILE = "src/shaders/vert.glsl";

int DEFAULT_NUM_PARTICLES = 200;

WorldState *world;

void init() {
  // Generate and bind the default VAO
  glGenVertexArrays(1, &g_default_vao);
  glBindVertexArray(g_default_vao);

  // Set the background color (RGBA)
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  // Your OpenGL settings, such as alpha, depth and others, should be
  // defined here! For the assignment, we only ask you to enable the
  // alpha channel.

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Init shaders. I let the program be global so I can delete it on exit.
  shader_program = util::loadShaders(VERT_FILE, FRAG_FILE);
  glUseProgram(shader_program);

  color_loc = glGetUniformLocation(shader_program, "uColor");
  if (color_loc == -1) {
    fprintf(stderr, "Error while getting uniform location");
  }

  MVP_loc = glGetUniformLocation(shader_program, "MVP");
  if (MVP_loc == -1) {
    fprintf(stderr, "Error while getting uniform location");
  }
  world->create_planets(world->particles, 1, 2, 0.3f,
                        glm::vec3(-3.0f, 0.0f, -5.0f),
                        glm::vec3(3.0f, 0.0f, -5.0f));
}

void release() {
  // Release the default VAO
  glDeleteVertexArrays(1, &g_default_vao);

  // Do not forget to release any memory allocation here!
  glDeleteProgram(shader_program);
}

void display(GLFWwindow *window) {
  // Clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Do mvp stuff
  // This needs to be here since we may revise the model/view/proj while running
  // Could cache it I guess, and only change on demand, but here is ok

  glm::mat4 V, P;
  auto ratio = (float)world->window.width / (float)world->window.height;
  auto &c = world->cam;

  V = glm::lookAt(c.pos, c.dir + c.pos, c.up);
  P = glm::perspective(glm::radians(c.fov), ratio, 0.1f, 10000.0f);

  for (const auto &p : world->particles) {
    glm::mat4 M;
    M = glm::translate(M, p.pos);

    glm::mat4 MVP = P * V * M;
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm::value_ptr(MVP));

    // Render a sphere
    glUniform4f(color_loc, 0.5, 0.2, 0.0, 0.5);
    agp::glut::glutSolidSphere(0.5f, 16, 8);
    glUniform4f(color_loc, 0.7, 0.7, 0.7, 1.0);
    agp::glut::glutWireSphere(0.5f, 16, 8);
  }

  // Swap buffers and force a redisplay
  glfwSwapBuffers(window);
  glfwPollEvents();
}

void resize_callback_h(GLFWwindow *win, int width, int height) {
  resize_callback(win, width, height, world);
}
void keyboard_callback_h(GLFWwindow *win, int key, int, int action, int) {
  keyboard_callback(win, key, action, world);
}
void cursor_callback_h(GLFWwindow *win, double xpos, double ypos) {
  cursor_callback(win, xpos, ypos, world);
}

int main(int argc, char **argv) {
  world = new WorldState(DEFAULT_NUM_PARTICLES);

  GLFWwindow *window = NULL;

  // Initialize GLFW
  if (!glfwInit()) {
    return GL_INVALID_OPERATION;
  }

  // Setup the OpenGL context version
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open the window and create the context
  window = glfwCreateWindow(world->window.width, world->window.height,
                            "Applied GPU Programming", NULL, NULL);

  if (window == NULL) {
    fprintf(stderr, "Could not create window");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // Capture the input events (e.g., keyboard)
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetKeyCallback(window, keyboard_callback_h);
  glfwSetCursorPosCallback(window, cursor_callback_h);
  glfwSetWindowSizeCallback(window, resize_callback_h);

  // Init GLAD to be able to access the OpenGL API
  if (!gladLoadGL()) {
    return GL_INVALID_OPERATION;
  }

  // Display OpenGL information
  util::displayOpenGLInfo();

  // Initialize the 3D view
  init();

  // Launch the main loop for rendering
  float dt = 0.017;
  float t = 0.0f;
  world->update(t, t); // Ensure initialized
  while (!glfwWindowShouldClose(window)) {
    update_held(world, dt);
    t += dt;
    world->update(dt, t);
    display(window);
  }

  // Release all the allocated memory
  release();

  // Release GLFW
  glfwDestroyWindow(window);
  glfwTerminate();

  delete world;

  return 0;
}

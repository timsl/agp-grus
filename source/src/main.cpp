#include "common.hpp"
#include "input.hpp"
#include "kernel.cuh"
#include "state.hpp"
#include "util.hpp"

using namespace std;
using namespace agp;

GLuint g_default_vao = 0;
GLint color_loc = -1;
GLint VP_loc = -1;
GLuint shader_program;

constexpr const char *FRAG_FILE = "src/shaders/frag.glsl";
constexpr const char *VERT_FILE = "src/shaders/vert.glsl";

int DEFAULT_NUM_PARTICLES = 4000;

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

  VP_loc = glGetUniformLocation(shader_program, "VP");
  if (VP_loc == -1) {
    fprintf(stderr, "Error while getting uniform location");
  }

  world->create_planets(world->particles, 3185.5f, 6371.0f, 0.3f,
                        glm::vec3(3185.5f, 0.0f, 0.0f),
                        glm::vec3(-3185.5f, 0.0f, 0.0f));

  // Send colors to opengl
  {
    std::vector<GLfloat> colorvec(16);
    for (int i = 0; i < 4; ++i) {
      auto &c = world->colors[i];
      colorvec[i * 4 + 0] = c.x;
      colorvec[i * 4 + 1] = c.y;
      colorvec[i * 4 + 2] = c.z;
      colorvec[i * 4 + 3] = c.w;
    }
    glUniform4fv(color_loc, 4, colorvec.data());
  }

  world->sphere =
    new Sphere(188.39f, 16, 8, 1, DEFAULT_NUM_PARTICLES, shader_program);
  world->gpu.init(reinterpret_cast<const CUParticle *>(world->particles.data()),
                  world->particles.size(), world->sphere->vbo_instanced);
}

void release() {
  world->gpu.clean();
  delete world->sphere;

  // Release the default VAO
  glDeleteVertexArrays(1, &g_default_vao);

  // Do not forget to release any memory allocation here!
  glDeleteProgram(shader_program);
}

void display(GLFWwindow *window) {
  // Clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // get the sphere object from world
  Sphere *sphere_object = world->sphere;

  // Do mvp stuff
  // This needs to be here since we may revise the model/view/proj while running
  // Could cache it I guess, and only change on demand, but here is ok

  glm::mat4 V, P;
  auto ratio = (float)world->window.width / (float)world->window.height;
  auto &c = world->cam;

  V = glm::lookAt(c.pos, c.dir + c.pos, c.up);
  P = glm::perspective(glm::radians(c.fov), ratio, 1.0f, 10000000.0f);
  glUniformMatrix4fv(VP_loc, 1, GL_FALSE, glm::value_ptr(P * V));


  sphere_object->prepare_render();

  // auto iter = sphere_object->particle_vbo_buffer;
  // for (const auto &p : world->particles) {
  //   glm::mat4 M = glm::translate(p.pos);
  //   util::storeModelViewMatrix(M, iter);
  //   util::storeByte(p.type, iter);
  //   iter = (char *)iter + sphere_object->data_length;
  // }

  // util::updateVbo(sphere_object->vbo_instanced,
  //                 sphere_object->particle_vbo_buffer,
  //                 sphere_object->data_length * DEFAULT_NUM_PARTICLES);

  sphere_object->render();
  sphere_object->finish_render();

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
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
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
  while (!glfwWindowShouldClose(window)) {
    update_held(world, dt);
    t += dt;
    if (world->held.simulation_running) {
      update(world, 5.8);
    }
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

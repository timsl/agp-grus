
#include "common.hpp"
#include "sphere.hpp"
#include "util.hpp"
#include "state.hpp"

using namespace std;
using namespace glm;
using namespace agp;
using namespace agp::glut;

GLuint g_default_vao = 0;
GLint color_loc = -1;
GLint MVP_loc = -1;
int width = 1280;
int height = 720;

constexpr const char *FRAG_FILE = "src/shaders/frag.glsl";
constexpr const char *VERT_FILE = "src/shaders/vert.glsl";

GLuint shader_program;

int NUM_PARTICLES = 200;

WorldState world(NUM_PARTICLES);

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
  world.create_planet(world.particles, 50, 150, 1, 2, glm::vec3(0.0f, 0.0f,-5.0f));
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

  glm::vec3 camPos(0.0f, 0.0f, 2.0f);
  camPos = glm::rotate(camPos, glm::radians(world.cam.angle),
                       glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 proj = glm::perspective(
      glm::radians(world.cam.fov), (float)width / (float)height, 0.1f, 1000.0f);

  for (const auto &p : world.particles) {
    glm::mat4 model;
    model = glm::translate(model, p.pos);

    glm::mat4 MVP = proj * view * model;
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm::value_ptr(MVP));

    // Render a sphere
    glUniform4f(color_loc, 0.5, 0.2, 0.0, 0.5);
    glutSolidSphere(0.5f, 16, 8);
    glUniform4f(color_loc, 0.7, 0.7, 0.7, 1.0);
    glutWireSphere(0.5f, 16, 8);
  }

  // Swap buffers and force a redisplay
  glfwSwapBuffers(window);
  glfwPollEvents();
}

void resize_callback(GLFWwindow *win, int width, int height) {
  ::width = width;
  ::height = height;
  glViewport(0, 0, width, height);
}

void keyboard_callback(GLFWwindow *win, int key, int, int action, int) {
  // Not handling release just means the code below is a bit neater,
  // since I'm not using it for any of the current buttons
  if (action == GLFW_RELEASE)
    return;

  switch (key) {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(win, 1);
    break;
  case GLFW_KEY_R:
    world.cam.angle = 0.0f;
    world.cam.fov = 90.0f;
    break;
  case GLFW_KEY_LEFT:
    world.cam.angle -= 5.0f;
    break;
  case GLFW_KEY_RIGHT:
    world.cam.angle += 5.0f;
    break;
  case GLFW_KEY_SLASH: // Where i expect + to be on swedish keyboard
    world.cam.fov += 2.0f;
    break;
  case GLFW_KEY_MINUS:
    world.cam.fov -= 2.0f;
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

int main(int argc, char **argv) {
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
  window =
      glfwCreateWindow(width, height, "Applied GPU Programming", NULL, NULL);

  if (window == NULL) {
    fprintf(stderr, "Could not create window");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // Capture the input events (e.g., keyboard)
  glfwSetKeyCallback(window, keyboard_callback);
  // glfwSetInputMode( ... );

  // Get window resizes
  glfwSetWindowSizeCallback(window, resize_callback);

  // Init GLAD to be able to access the OpenGL API
  if (!gladLoadGL()) {
    return GL_INVALID_OPERATION;
  }

  // Display OpenGL information
  util::displayOpenGLInfo();

  // Initialize the 3D view
  init();

  // Launch the main loop for rendering
  float dt = 1.0f;
  float t = 0.0f;
  world.update(t, t);           // Ensure initialized
  while (!glfwWindowShouldClose(window)) {
    t += dt;
    world.update(dt, t);
    display(window);
  }

  // Release all the allocated memory
  release();

  // Release GLFW
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

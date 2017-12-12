#include "common.hpp"
#include "util.hpp"

using namespace std;
using namespace agp;

BYTE *readFile(const char *filename) {
  FILE *file = NULL;
  BYTE *data = NULL;
  long length = 0;

  file = fopen(filename, "rb");

  if (!file) {
    fprintf(stderr, "File %s not found!\n", filename);
    return NULL;
  }

  // Calculate the length
  fseek(file, 0, SEEK_END);
  length = ftell(file);
  fseek(file, 0, SEEK_SET);

  // Read the content of the file
  data = (BYTE *)malloc(sizeof(BYTE) * (length + 1));
  fread(data, length, sizeof(BYTE), file);
  data[length] = '\0';

  fclose(file);

  return data;
}

GLuint createShader(GLenum shader_type, const char *shader_filename) {
  GLuint shader = 0;
  BYTE *shader_data = NULL;
  GLint hr = GL_TRUE;
  int length = 0;

  // Create the shader and read the code from the file
  shader = glCreateShader(shader_type);
  shader_data = readFile(shader_filename);

  // Compile the shader
  glShaderSource(shader, 1, (const GLchar **)&shader_data, NULL);
  glCompileShader(shader);

  // Check if the Vertex Shader compiled successfully
  glGetShaderiv(shader, GL_COMPILE_STATUS, &hr);
  if (hr == GL_FALSE) {
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

    vector<char> error_string(length + 1);
    glGetShaderInfoLog(shader, length, NULL, &error_string[0]);

    fprintf(stderr, "Error compiling the shader:\n%s\n", &error_string[0]);
  }

  free(shader_data);

  return shader;
}

GLuint util::loadShaders(const char *vertex_shader_filename,
                         const char *fragment_shader_filename) {
  GLuint program = 0;
  GLuint vertex_shader = 0;
  GLuint fragment_shader = 0;
  GLint hr = GL_TRUE;
  int length = 0;

  // Create the Vertex and Fragment Shaders
  vertex_shader = createShader(GL_VERTEX_SHADER, vertex_shader_filename);
  fragment_shader = createShader(GL_FRAGMENT_SHADER, fragment_shader_filename);

  // Create the program, attach the shaders and link it
  program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);

  // Bind the vertex position to index 0
  // glBindAttribLocation(program, 0, "vertexPosition");

  // Link the OpenGL program
  glLinkProgram(program);

  // Check if the program is linked
  glGetShaderiv(program, GL_LINK_STATUS, &hr);
  if (hr == GL_FALSE) {
    glGetShaderiv(program, GL_INFO_LOG_LENGTH, &length);

    vector<char> error_string(length + 1);
    glGetProgramInfoLog(program, length, NULL, &error_string[0]);

    fprintf(stderr, "Error linking the program:\n%s\n", &error_string[0]);
  }

  // Release all the memory
  glDetachShader(program, vertex_shader);
  glDetachShader(program, fragment_shader);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  return program;
}

void util::displayOpenGLInfo() {
  // Display information about the GPU and OpenGL version
  printf("OpenGL v%d.%d\n", GLVersion.major, GLVersion.minor);
  printf("Vendor: %s\n", glGetString(GL_VENDOR));
  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("Version: %s\n", glGetString(GL_VERSION));
  printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
}

GLuint util::createEmptyVbo(int nr_floats) {
  GLuint vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, nr_floats * sizeof(GL_FLOAT), NULL,
               GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return vbo;
}

void util::addInstancedAttribute(GLuint vao, GLuint vbo, int attribute,
                                 int dataSize, GLsizei stride, size_t offset,
                                 bool integral) {
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  if (integral) {
    glVertexAttribIPointer(attribute, dataSize, GL_UNSIGNED_BYTE, stride,
                           (GLvoid *)(offset));
  } else {
    glVertexAttribPointer(attribute, dataSize, GL_FLOAT, GL_FALSE, stride,
                          (GLvoid *)(offset));
  }

  glVertexAttribDivisor(attribute, 1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void util::updateVbo(GLuint vbo, void *data, int nr) {
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, nr, NULL, GL_STREAM_DRAW);
  glBufferSubData(GL_ARRAY_BUFFER, 0, nr, data);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void util::storeModelViewMatrix(glm::mat4 MV, void *to_ptr) {
  auto *floats_ptr = glm::value_ptr(MV[3]);
  std::copy(floats_ptr, floats_ptr + 4, (float*)to_ptr);
}

void util::storeByte(char b, void *to_ptr) {
  *((char*)to_ptr + 16) = b;
}

void util::bindAttrib(GLuint program, int attribute, char *variable_name) {
  glBindAttribLocation(program, attribute, variable_name);
}

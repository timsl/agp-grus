#ifndef _AGP_UTIL_H
#define _AGP_UTIL_H

namespace agp {
namespace util {
/**
 * Helper method that loads the Vertex and Fragment Shaders, and
 * returns the OpenGL Program associated with them.
 */
GLuint loadShaders(const char *vertex_shader_filename,
                   const char *fragment_shader_filename);

/**
 * Helper method that displays information about OpenGL.
 */
void displayOpenGLInfo();

/**
 * Helper method that generates the vbo for the instanced rendering
 */
GLuint createEmptyVbo(int nr_floats);

/**
 * Helper method that adds an instanced attribute to a vao
 */
void addInstancedAttribute(GLuint vao, GLuint vbo, int attribute, int dataSize,
                           int instancedDataLength, int offset);

/**
 * Helper method to update our vbo for instanced rendering
 */
void updateVbo(GLuint vbo, float* data, int nr_floats);

void storeModelViewMatrix(glm::mat4 MV, std::vector<float>::iterator iter);

} // namespace util
} // namespace agp

#endif

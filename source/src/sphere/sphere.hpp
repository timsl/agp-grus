#ifndef _AGP_SPHERE_H
#define _AGP_SPHERE_H

#include "state.hpp"
#include "util.hpp"

struct Sphere {
  GLuint vao_sphere;
  GLuint vbo_vertices;
  GLuint vbo_indices;
  GLuint vbo_instanced;
  GLfloat radius;
  GLuint slices;
  GLuint stacks;
  GLint nr_vertices;
  GLint nr_spheres;
  GLint nVertIdxsPerPart;
  size_t element_size;
  void *particle_vbo_buffer;
  unsigned data_length;

  Sphere(GLfloat radius, GLint slices, GLint stacks, int n,
         GLint n_particles, GLuint program);
  void generate_sphere(GLfloat radius, GLint slices, GLint stacks,
                       GLfloat *verticies, GLfloat *normals);
  void prepare_render();
  void render();
  void finish_render();
  void setCircleTable(GLfloat **sint, GLfloat **cost, const int n,
                      const GLboolean halfCircle);
  void clean_up();
};

#endif

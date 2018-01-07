#ifndef _AGP_SPHERE_H
#define _AGP_SPHERE_H

#include "common.hpp"
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
  // void *particle_vbo_buffer;
  unsigned data_length;

  // Constructor, sets up the sphere verticies and the VAO and VBOs
  Sphere(GLfloat radius, GLint slices, GLint stacks, int n, GLint n_particles,
         GLuint program);
  ~Sphere();
  // Helper function for generating the sphere verticies
  void generate_sphere(GLfloat radius, GLint slices, GLint stacks,
                       GLfloat *verticies, GLfloat *normals);
  // Enables everything needed for rendering
  void prepare_render();
  // Renders the spheres
  void render();
  // Disables everything needed for rendering 
  void finish_render();
  // Helper function for generating the sphere
  void setCircleTable(GLfloat **sint, GLfloat **cost, const int n,
                      const GLboolean halfCircle);
};

#endif

#ifndef _AGP_SPHERE_H
#define _AGP_SPHERE_H

#include "state.hpp"
#include "util.hpp"

struct Sphere {
    GLuint vbo_vertices;
    GLuint vbo_instanced;
    GLfloat radius;
    GLuint slices;
    GLuint stacks;
    GLuint nr_vertices;
    std::vector<float> particle_vbo_buffer;
    int data_length;

    Sphere(GLfloat radius, GLint slices, GLint stacks, int n, GLuint vao, int n_particles);
    void generate_sphere(GLfloat radius, GLint slcies, GLint stacks, GLfloat *verticies, GLfloat *normals); 
    void render(GLuint vao);
};


namespace agp {
namespace glut {
 /**
 * Helper methods that simulate the GLUT functionality to draw
 * a solid and wired sphere, respectively. The code is based on
 * the FreeGLUT implementation, but with some modifications.
 */

void glutSolidSphere(GLfloat radius, GLint slices, GLint stacks);
void glutSolidSphereInstanced(GLfloat radius, GLint slices, GLint stacks,
                              GLint nr_spheres);
void glutWireSphere(GLfloat radius, GLint slices, GLint stacks);

 /*
  * Methods for setting up a sphere
  */

} // namespace glut
} // namespace agp

#endif

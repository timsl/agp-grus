
#include "common.hpp"
#include "sphere.hpp"

using namespace std;
using namespace agp;

Sphere::Sphere(GLfloat radius, GLint slices, GLint stacks, int n, GLuint vao,
               GLint n_particles, GLuint program) {

  this->radius = radius;
  this->slices = slices;
  this->stacks = stacks;
  this->nr_spheres = n_particles;
  this->data_length = 17;
  this->particle_vbo_buffer.reserve(nr_spheres * data_length);

  for (size_t i = 0; i < nr_spheres * data_length; ++i) {
    particle_vbo_buffer.push_back(0.0f);
  }

  GLfloat *vertices = NULL;
  GLfloat *normals = NULL;
  GLushort *stripIdx = NULL;
  GLint nVert = slices * (stacks - 1) + 2;
  GLint nVertIdxsPerPart = (slices + 1) * 2;
  //    GLuint   vbo_normals      = 0;
  GLuint ibo_elements = 0;
  GLint idx = 0;
  GLushort offset = 0;
  GLsizei numVertIdxs = stacks * nVertIdxsPerPart;

  if (slices == 0 || stacks < 2) {
    return;
  }

  // Allocate vertex and normal buffers
  vertices = (GLfloat *)malloc(nVert * 3 * sizeof(GLfloat));
  normals = (GLfloat *)malloc(nVert * 3 * sizeof(GLfloat));

  // Generate vertices and normals
  generate_sphere(radius, slices, stacks, vertices, normals);

  // create vbo for sphere
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo_vertices);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);

  glVertexAttribPointer(0,        // attribute
                        3,        // number of elements per vertex, here (x,y,z)
                        GL_FLOAT, // the type of each element
                        GL_FALSE, // take our values as-is
                        0,        // no extra data between each position
                        0         // offset of first element
  );
  glBufferData(GL_ARRAY_BUFFER, nVert * 3 * sizeof(vertices[0]), vertices,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);

  nr_vertices = nVert;
  // Create the vbo used for the instanced rendering
  vbo_instanced = util::createEmptyVbo(nr_spheres * data_length);

  util::addInstancedAttribute(vao, vbo_instanced, 1, 4, data_length, 0);
  util::addInstancedAttribute(vao, vbo_instanced, 2, 4, data_length, 4);
  util::addInstancedAttribute(vao, vbo_instanced, 3, 4, data_length, 8);
  util::addInstancedAttribute(vao, vbo_instanced, 4, 4, data_length, 12);
  util::addInstancedAttribute(vao, vbo_instanced, 5, 1, data_length, 16);
  
  // bind the attributes to the shader
  
  util::bindAttrib(program, 0, "pos");
  util::bindAttrib(program, 1, "M");
  util::bindAttrib(program, 5, "type");

  // Release the allocations and buffers
  free(stripIdx);
  free(normals);
  free(vertices);
}

void Sphere::prepare_render(GLuint vao) {
  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);
  glEnableVertexAttribArray(3);
  glEnableVertexAttribArray(4);
  glEnableVertexAttribArray(5);
}

void Sphere::render() {
  glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, nr_vertices, nr_spheres);
  // glDrawArrays(GL_TRIANGLE_STRIP, 0, nr_vertices);
}

void Sphere::finish_render() {
  glDisableVertexAttribArray(5);
  glDisableVertexAttribArray(4);
  glDisableVertexAttribArray(3);
  glDisableVertexAttribArray(2);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
}

void Sphere::clean_up() {
  glDeleteBuffers(1, &vbo_vertices);
  glDeleteBuffers(1, &vbo_instanced);
}

void setCircleTable(GLfloat **sint, GLfloat **cost, const int n,
                    const GLboolean halfCircle) {
  // Table size, the sign of n flips the circle direction
  const int size = abs(n);

  // Determine the angle between samples
  const GLfloat angle =
      (halfCircle ? 1 : 2) * (GLfloat)M_PI / (GLfloat)((n == 0) ? 1 : n);

  // Allocate memory for n samples, plus duplicate of first entry at the end
  *sint = (GLfloat *)malloc(sizeof(GLfloat) * (size + 1));
  *cost = (GLfloat *)malloc(sizeof(GLfloat) * (size + 1));

  // Compute cos and sin around the circle
  (*sint)[0] = 0.0;
  (*cost)[0] = 1.0;

  for (int i = 1; i < size; i++) {
    (*sint)[i] = (GLfloat)sin(angle * i);
    (*cost)[i] = (GLfloat)cos(angle * i);
  }

  if (halfCircle) {
    (*sint)[size] = 0.0f;  // sin PI
    (*cost)[size] = -1.0f; // cos PI
  } else {
    // Last sample is duplicate of the first (sin or cos of 2 PI)
    (*sint)[size] = (*sint)[0];
    (*cost)[size] = (*cost)[0];
  }
}
void generateSphere(GLfloat radius, GLint slices, GLint stacks,
                    GLfloat *vertices, GLfloat *normals) {
  int idx = 0; // idx into vertex / normal buffer
  GLfloat x = 0.0f;
  GLfloat y = 0.0f;
  GLfloat z = 0.0f;
  GLfloat *sint1 = NULL;
  GLfloat *cost1 = NULL;
  GLfloat *sint2 = NULL;
  GLfloat *cost2 = NULL;

  // precompute values on unit circle
  setCircleTable(&sint1, &cost1, -slices, GL_FALSE);
  setCircleTable(&sint2, &cost2, stacks, GL_TRUE);

  // Top
  vertices[0] = 0.f;
  vertices[1] = 0.f;
  vertices[2] = radius;
  normals[0] = 0.f;
  normals[1] = 0.f;
  normals[2] = 1.f;
  idx = 3;

  // Each stack
  for (int i = 1; i < stacks; i++) {
    for (int j = 0; j < slices; j++, idx += 3) {
      x = cost1[j] * sint2[i];
      y = sint1[j] * sint2[i];
      z = cost2[i];

      vertices[idx] = x * radius;
      vertices[idx + 1] = y * radius;
      vertices[idx + 2] = z * radius;
      normals[idx] = x;
      normals[idx + 1] = y;
      normals[idx + 2] = z;
    }
  }

  // Bottom
  vertices[idx] = 0.f;
  vertices[idx + 1] = 0.f;
  vertices[idx + 2] = -radius;
  normals[idx] = 0.f;
  normals[idx + 1] = 0.f;
  normals[idx + 2] = -1.f;

  // Done creating vertices, release sin and cos tables
  free(sint1);
  free(cost1);
  free(sint2);
  free(cost2);
}

void Sphere::generate_sphere(GLfloat radius, GLint slices, GLint stacks,
                             GLfloat *vertices, GLfloat *normals) {
  int idx = 0; // idx into vertex / normal buffer
  GLfloat x = 0.0f;
  GLfloat y = 0.0f;
  GLfloat z = 0.0f;
  GLfloat *sint1 = NULL;
  GLfloat *cost1 = NULL;
  GLfloat *sint2 = NULL;
  GLfloat *cost2 = NULL;

  // precompute values on unit circle
  setCircleTable(&sint1, &cost1, -slices, GL_FALSE);
  setCircleTable(&sint2, &cost2, stacks, GL_TRUE);

  // Top
  vertices[0] = 0.f;
  vertices[1] = 0.f;
  vertices[2] = radius;
  normals[0] = 0.f;
  normals[1] = 0.f;
  normals[2] = 1.f;
  idx = 3;

  // Each stack
  for (int i = 1; i < stacks; i++) {
    for (int j = 0; j < slices; j++, idx += 3) {
      x = cost1[j] * sint2[i];
      y = sint1[j] * sint2[i];
      z = cost2[i];

      vertices[idx] = x * radius;
      vertices[idx + 1] = y * radius;
      vertices[idx + 2] = z * radius;
      normals[idx] = x;
      normals[idx + 1] = y;
      normals[idx + 2] = z;
    }
  }

  // Bottom
  vertices[idx] = 0.f;
  vertices[idx + 1] = 0.f;
  vertices[idx + 2] = -radius;
  normals[idx] = 0.f;
  normals[idx + 1] = 0.f;
  normals[idx + 2] = -1.f;

  // Done creating vertices, release sin and cos tables
  free(sint1);
  free(cost1);
  free(sint2);
  free(cost2);
}

void glut::glutSolidSphere(GLfloat radius, GLint slices, GLint stacks) {
  GLfloat *vertices = NULL;
  GLfloat *normals = NULL;
  GLushort *stripIdx = NULL;
  GLint nVert = slices * (stacks - 1) + 2;
  GLint nVertIdxsPerPart = (slices + 1) * 2;
  GLuint vbo_coords = 0;
  //    GLuint   vbo_normals      = 0;
  GLuint ibo_elements = 0;
  GLint idx = 0;
  GLushort offset = 0;
  GLsizei numVertIdxs = stacks * nVertIdxsPerPart;

  if (slices == 0 || stacks < 2) {
    return;
  }

  // Allocate vertex and normal buffers
  vertices = (GLfloat *)malloc(nVert * 3 * sizeof(GLfloat));
  normals = (GLfloat *)malloc(nVert * 3 * sizeof(GLfloat));

  // Generate vertices and normals
  generateSphere(radius, slices, stacks, vertices, normals);

  // Allocate buffers for indices
  stripIdx = (GLushort *)malloc((slices + 1) * 2 * stacks * sizeof(GLushort));

  // Generate vertex index arrays for drawing with glDrawElements (all
  // stacks, including top / bottom, are covered with a triangle strip)
  {
    // Top stack
    for (int j = 0; j < slices; j++, idx += 2) {
      stripIdx[idx] = j + 1; // 0 is top vertex, 1 is first for
      stripIdx[idx + 1] = 0; // first stack
    }
    stripIdx[idx] = 1; // Repeat first slice's idx for closing off shape
    stripIdx[idx + 1] = 0;
    idx += 2;

    // Middle stacks (strip indices are relative to first index belonging
    // to strip, NOT relative to first vertex/normal pair in array)
    for (int i = 0; i < stacks - 2; i++, idx += 2) {
      // triangle_strip indices start at 1 (0 is top vertex), and we
      // advance one stack down as we go along
      offset = 1 + i * slices;
      for (int j = 0; j < slices; j++, idx += 2) {
        stripIdx[idx] = offset + j + slices;
        stripIdx[idx + 1] = offset + j;
      }
      stripIdx[idx] = offset + slices; // Repeat first slice's idx
      stripIdx[idx + 1] = offset;      // for closing off shape
    }

    // Bottom stack (triangle_strip indices start at 1, with 0 top vertex,
    // and we advance one stack down as we go along)
    offset = 1 + (stacks - 2) * slices;
    for (int j = 0; j < slices; j++, idx += 2) {
      stripIdx[idx] = nVert - 1;      // Zero-based index, last element in
      stripIdx[idx + 1] = offset + j; // array (bottom vertex)
    }
    stripIdx[idx] = nVert - 1;  // Repeat first slice's idx for closing
    stripIdx[idx + 1] = offset; // off shape
  }

  // Generate the buffer object for the vertices
  glGenBuffers(1, &vbo_coords);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_coords);
  glBufferData(GL_ARRAY_BUFFER, nVert * 3 * sizeof(vertices[0]), vertices,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Generate the buffer object for the normals
  //    glGenBuffers(1, &vbo_normals);
  //    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
  //    glBufferData(GL_ARRAY_BUFFER, nVert * 3 * sizeof(normals[0]),
  //                 normals, GL_STATIC_DRAW);
  //    glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Generate the buffer object for the indices
  glGenBuffers(1, &ibo_elements);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, numVertIdxs * sizeof(stripIdx[0]),
               stripIdx, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // Render the sphere
  {
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_coords);
    glVertexAttribPointer(0, // attribute
                          3, // number of elements per vertex, here (x,y,z)
                          GL_FLOAT, // the type of each element
                          GL_FALSE, // take our values as-is
                          0,        // no extra data between each position
                          0         // offset of first element
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //        glEnableVertexAttribArray(attribute_v_normal);
    //        glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
    //        glVertexAttribPointer(
    //            attribute_v_normal, // attribute
    //            3,                  // number of elements per vertex, here
    //            (x,y,z) GL_FLOAT,           // the type of each element
    //            GL_FALSE,           // take our values as-is
    //            0,                  // no extra data between each position
    //            0                   // offset of first element
    //        );
    //        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);

    if (stacks > 1) {
      const size_t size = sizeof(stripIdx[0]) * nVertIdxsPerPart;

      for (int i = 0; i < stacks; i++) {
        glDrawElements(GL_TRIANGLE_STRIP, nVertIdxsPerPart, GL_UNSIGNED_SHORT,
                       (GLvoid *)(size * i));
      }
    } else {
      glDrawElements(GL_TRIANGLE_STRIP, nVertIdxsPerPart, GL_UNSIGNED_SHORT,
                     NULL);
    }

    // Clean existing bindings
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(0);
    //        glDisableVertexAttribArray(attribute_v_normal);
  }

  // Release the allocations and buffers
  free(stripIdx);
  free(normals);
  free(vertices);

  glDeleteBuffers(1, &vbo_coords);
  //        glDeleteBuffers(1, &vbo_normals);
  glDeleteBuffers(1, &ibo_elements);
}

void glut::glutWireSphere(GLfloat radius, GLint slices, GLint stacks) {
  GLint polygonMode[2];

  // Set the polygon mode
  glGetIntegerv(GL_POLYGON_MODE, polygonMode);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  glutSolidSphere(radius, slices, stacks);

  // Reset the polygon mode to the old value
  glPolygonMode(GL_FRONT_AND_BACK, polygonMode[0]);
}

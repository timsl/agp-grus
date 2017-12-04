#ifndef _AGP_SPHERE_H
#define _AGP_SPHERE_H

namespace agp
{
    namespace glut
    {
        /**
         * Helper methods that simulate the GLUT functionality to draw
         * a solid and wired sphere, respectively. The code is based on
         * the FreeGLUT implementation, but with some modifications.
         */
        void glutSolidSphere(GLfloat radius, GLint slices, GLint stacks);
        void glutWireSphere(GLfloat radius, GLint slices, GLint stacks);
    }
}

#endif


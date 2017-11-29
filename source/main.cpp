#include "common.hpp"
#include "sphere.hpp"
#include "util.hpp"

#include <iostream>
#include <math.h>

#include <vector>
#include <random>

#define PI 4*atan(1)
#define NUM_PARTICLES 50
#define D_ROT 3.6*(PI / 180.0f)
#define D_MOVE 0.5f
#define SENSE 0.1*(PI/180.0f)
#define INVERT_Y -1

using namespace std;
using namespace glm;
using namespace agp;
using namespace agp::glut;

GLuint g_default_vao = 0;
unsigned int shaderProgram;

float fov = 60.0f;
double old_xpos = 0, old_ypos = 0;
vec3 cam_pos(  0.0f, 0.0f,  0.0f);
vec3 cam_dir(  0.0f, 0.0f,  1.0f);
vec3 cam_up(   0.0f, 1.0f,  0.0f);


float rand_float(float lim) {
    return (rand() / (float)RAND_MAX)*2*lim - lim;
}


typedef struct Particle {
    vec3 pos;
    vec3 ori;
    float rad;
} Particle;

vector<Particle> particles;

void move(vec3 dir){
    cam_pos += dir;
    //cam_dir += dir;
    //cam_up  += dir;
}

// angle in radians
void rotate(vec3 axis, float angle){
    mat4 R = glm::rotate(angle, axis);
    cam_dir = vec3(R * vec4(cam_dir, 1.0f));
    cam_up  = vec3(R * vec4(cam_up,  1.0f));
}

void init() {
    // Generate and bind the default VAO
    glGenVertexArrays(1, &g_default_vao);
    glBindVertexArray(g_default_vao);

    // Set the background color (RGBA)
    glClearColor(0.17f, 0.17f, 0.17f, 0.0f);


    // load shader program and enable alpha
    shaderProgram = agp::util::loadShaders("vs.glsl", "fs.glsl");
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //init particle centers:
    float lim = 2.5f;
    for (int i = 0; i < NUM_PARTICLES; ++i){
                Particle p;
                p.pos = vec3(rand_float(lim), rand_float(lim), rand_float(lim));
                p.ori = vec3(rand_float(1), rand_float(1), rand_float(1));
                p.rad = rand_float(0.75f);
        particles.push_back(p);
    }
}

void release() {
    // Release the default VAO
    glDeleteVertexArrays(1, &g_default_vao);

    // Do not forget to release any memory allocation here!
}

void display(GLFWwindow *window) {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //printf("GLFW triggered the display() callback!\n");

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Your rendering code must be here!
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    glUseProgram(shaderProgram);
    glBindVertexArray(g_default_vao);


    glm::mat4 M(1), V, P;
    V = glm::lookAt(cam_pos, cam_dir+cam_pos, cam_up);
    P = glm::perspective(glm::radians(fov), (float) w / (float) h, 0.1f, 100.0f);
    for (auto p : particles) {
        M = glm::translate(M, p.pos);
        M *= glm::rotate(p.ori.x, vec3(1,0,0));
        M *= glm::rotate(p.ori.y, vec3(0,1,0));
        M *= glm::rotate(p.ori.z, vec3(0,0,1));
        unsigned int mvp = glGetUniformLocation(shaderProgram, "MVP");
        glUniformMatrix4fv(mvp, 1, false, glm::value_ptr(P*V*M));
        glutWireSphere(p.rad, 8, 8);
        glutSolidSphere(p.rad, 8, 8);
    }

    // Swap buffers and force a redisplay
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos){
    double x_diff = old_xpos - xpos;
    old_xpos = xpos;
    double y_diff = old_ypos - ypos;
    old_ypos = ypos;
    rotate(glm::cross(cam_up, cam_dir), y_diff * SENSE * INVERT_Y);
    rotate(cam_up, x_diff * SENSE);
#if DEBUG
    cout << "rotating   " << x_diff << "     "<<  y_diff << endl;
#endif
}


void keypress_cb(GLFWwindow *window, int key, int scancode, int action, int mods){
    //if (action == GLFW_PRESS) {
        //cout << "key pressed: " << key << endl;
        switch(key){
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;

            case GLFW_KEY_LEFT:
                rotate(cam_up, D_ROT);
                break;
            case GLFW_KEY_RIGHT:
                rotate(cam_up, -D_ROT);
                break;
            case GLFW_KEY_UP:
                rotate(glm::cross(cam_up, cam_dir), -D_ROT);
                break;
            case GLFW_KEY_DOWN:
                rotate(glm::cross(cam_up, cam_dir), D_ROT);
                break;


            case GLFW_KEY_W:
                move(vec3( D_MOVE * glm::normalize(cam_dir)));
                break;
            case GLFW_KEY_S:
                move(vec3(-D_MOVE * glm::normalize(cam_dir)));
                break;
            case GLFW_KEY_A:
                move(vec3( D_MOVE * glm::normalize(glm::cross(cam_up, cam_dir))));
                break;
            case GLFW_KEY_D:
                move(vec3(-D_MOVE * glm::normalize(glm::cross(cam_up, cam_dir))));
                break;
            case GLFW_KEY_Q:

                move(vec3( D_MOVE * glm::normalize(cam_up)));
                break;
            case GLFW_KEY_Z:
                move(vec3(-D_MOVE * glm::normalize(cam_up)));
                break;

                //plus
            case 47:
                fov += 3;
                break;
                //minus
            case 45:
                fov -= 3;
                break;
        }
#if DEBUG
        cout << "\n\nup · dir:\t";
        cout << glm::dot(cam_up, cam_dir) << endl;
        cout << glm::dot(cam_up, cam_dir-cam_pos) << endl;
        cout << "pos\t" << cam_pos.x << " " << cam_pos.y << " " << cam_pos.z << endl;
        cout << "up\t"  << cam_up.x  << " " << cam_up.y  << " " << cam_up.z  << endl;
        cout << "dir\t" << cam_dir.x << " " << cam_dir.y << " " << cam_dir.z << endl;
#endif

    //}
}


int main(int argc, char **argv) {
    srand(4711);
    GLFWwindow *window = NULL;

    // Initialize GLFW
	if(!glfwInit()) {
		return GL_INVALID_OPERATION;
	}

    // Setup the OpenGL context version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open the window and create the context
    // my window manager makes windows named "float" floating
	window = glfwCreateWindow(800, 600, "float", NULL, NULL);

	if(window == NULL) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Capture the input events (e.g., keyboard)
     glfwSetKeyCallback( window, keypress_cb );
     glfwSetCursorPosCallback(window, cursor_pos_callback);
     glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	// glfwSetInputMode( ... );

    // Init GLAD to be able to access the OpenGL API
    if (!gladLoadGL()) {
        return GL_INVALID_OPERATION;
    }

    // Display OpenGL information
    util::displayOpenGLInfo();

    // Initialize the 3D view
    init();

    // Launch the main loop for rendering
    while (!glfwWindowShouldClose(window)) {
        display(window);
    }

    // Release all the allocated memory
    release();

    // Release GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

	return 0;
}


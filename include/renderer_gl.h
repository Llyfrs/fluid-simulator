#ifndef RENDERER_GL_H
#define RENDERER_GL_H

#include <GL/glew.h>
#include "data_structures.h" // For GLData, ParticleSystem
#include <GL/freeglut.h>
#include <string>


// Initialization and Resource Management
void renderer_init_gl(int argc, char** argv);
void renderer_prepare_gl_objects(); // Renamed from prepareGLObjects
void renderer_release_gl();         // Renamed from releaseOpenGL

// GLUT Callbacks - these will internally use g_glData, gh_particleSystem
void glut_display_callback();
void glut_reshape_callback(GLsizei w, GLsizei h);
void glut_idle_callback();

// Actual rendering and logic (callable by callbacks)
void renderer_display_scene();
void renderer_update_camera();
void renderer_handle_resize(GLsizei w, GLsizei h);
void renderer_simulation_and_vbo_update();

GLuint load_shaders_from_files(const std::string& vertex_filepath, const std::string& fragment_filepath);

#endif // RENDERER_GL_H
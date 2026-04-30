#include "renderer_gl.h"
#include "globals.h"      // For g_glData, gh_particleSystem
#include "cuda_interop.h" // For cuda_run_simulation_step
#include <cstdio>
#include <cmath>
#include "config.h"
#include <GL/glew.h> // MUST be included before other GL headers
#include <GL/freeglut.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void renderer_init_gl(int argc, char** argv) {
    glutInit(&argc, argv);
    // ... (rest of initGL, use g_glData.viewportWidth etc.)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(g_glData.viewportWidth, g_glData.viewportHeight);
    // ...
    glutCreateWindow("Particle System Renderer");

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        exit(1);
    }

    // Try a few common working directories (repo root vs build dir).
    g_glData.shaderProgramID = load_shaders_from_files("shaders/sphere.vert", "shaders/sphere.frag");
    if (g_glData.shaderProgramID == 0) {
        g_glData.shaderProgramID = load_shaders_from_files("../shaders/sphere.vert", "../shaders/sphere.frag");
    }
    if (g_glData.shaderProgramID == 0) {
        g_glData.shaderProgramID = load_shaders_from_files("../../shaders/sphere.vert", "../../shaders/sphere.frag");
    }
    printf("OpenGL initialized with shader program ID: %u\n", g_glData.shaderProgramID);

    // Initialize camera settings
    g_glData.cameraDistance = 10.0f;  // Distance from origin
    g_glData.cameraAngleXY = 45.0f;  // Rotation around Y axis
    g_glData.cameraAngleZ = 30.0f;   // Elevation angle

    // Register GLUT callbacks that call our wrapped functions
    glutDisplayFunc(glut_display_callback);
    glutReshapeFunc(glut_reshape_callback);
    glutIdleFunc(glut_idle_callback);

    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

}

// This is a COMPLETE REPLACEMENT for your existing function.
void renderer_prepare_gl_objects() {
    // Note: The shader should already be loaded by renderer_init_gl

    // --- Vertex Array Object (VAO) Setup ---
    // A VAO stores all the state needed to supply vertex data.
    glGenVertexArrays(1, &g_glData.vaoID);
    glBindVertexArray(g_glData.vaoID); // Start recording the setup for our particles

    // --- Vertex Buffer Object (VBO) Setup (same as your PBO logic) ---
    glGenBuffers(1, &g_glData.pboID);
    glBindBuffer(GL_ARRAY_BUFFER, g_glData.pboID);
    // Use the dynamic number of particles from g_glData, not a macro
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    printf("\nCreated OpenGL VBO (ID: %u) for %d particle positions.\n", g_glData.pboID, NUM_PARTICLES);

    // --- Link VBO to Shader Input ---
    // This is the most critical part. It tells OpenGL how to read the VBO data
    // and where to send it in the vertex shader.
    glEnableVertexAttribArray(0); // Corresponds to "layout(location = 0)" in sphere.vert
    glVertexAttribPointer(
            0,        // Attribute location 0
            3,        // Size of the vertex attribute (vec3 -> 3 floats)
            GL_FLOAT, // Type of the data
            GL_FALSE, // Should it be normalized?
            0,        // Stride (0 means data is tightly packed)
            (void*)0  // Offset to the first component
    );

    // --- Stop Recording State ---
    // Unbind the VAO. All the configuration (glBindBuffer, glEnableVertexAttribArray,
    // glVertexAttribPointer) is now saved to g_glData.vaoID.
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind the VBO
}

// This is a COMPLETE REPLACEMENT for your existing function.
void renderer_display_scene() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // --- Activate our shader program ---
    glUseProgram(g_glData.shaderProgramID);

    // --- Set up camera matrices using GLM ---
    // Using the fixed debug camera at (0,0,5). The dynamic camera calculation is commented out for now.

    float camX = g_glData.cameraDistance * cos(g_glData.cameraAngleZ * M_PI / 180.0f) * sin(g_glData.cameraAngleXY * M_PI / 180.0f);
    float camY = g_glData.cameraDistance * sin(g_glData.cameraAngleZ * M_PI / 180.0f);
    float camZ = g_glData.cameraDistance * cos(g_glData.cameraAngleZ * M_PI / 180.0f) * cos(g_glData.cameraAngleXY * M_PI / 180.0f);
    glm::vec3 cameraPos = glm::vec3(camX, camY, camZ);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);


    glm::mat4 viewMatrix = glm::lookAt(
        cameraPos,                // Camera position
        cameraTarget,             // Look at the origin
        glm::vec3(0.0f, 1.0f, 0.0f) // Up vector
    );

    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

    // --- Send matrix data to the shaders (as uniforms) ---
    // FIX: The uniform name in glGetUniformLocation now matches the shader.
    GLint viewLoc = glGetUniformLocation(g_glData.shaderProgramID, "viewMatrix");
    GLint projLoc = glGetUniformLocation(g_glData.shaderProgramID, "projectionMatrix");

    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    // --- Send other data to shaders ---
    GLint radiusLoc = glGetUniformLocation(g_glData.shaderProgramID, "pointRadius");
    GLint lightDirLoc = glGetUniformLocation(g_glData.shaderProgramID, "lightDir");
    GLint screenHeightLoc = glGetUniformLocation(g_glData.shaderProgramID, "screenHeight");

    // FIX: Use glUniform1f for screenHeight, as it's a float, not a matrix.
    glUniform1f(screenHeightLoc, (float)height);

    // FIX: Use a much smaller, more reasonable radius.
    glUniform1f(radiusLoc, GRID_SPACING / (3 *  PARTICLES_PER_CELL_AXIS)); // Adjusted radius for better visibility
    glUniform3f(lightDirLoc, 0.0f, -1.0f, 0.0f);

    // --- Draw the particles ---
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(g_glData.vaoID);

    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

    // --- Clean up ---
    glBindVertexArray(0);
    glUseProgram(0);

    glutSwapBuffers();
}


void renderer_handle_resize(GLsizei w, GLsizei h) {
    // Prevent division by zero if window is minimized
    if (h == 0) {
        h = 1;
    }

    // This is the only line that matters for the modern pipeline.
    // It tells OpenGL how to map its [-1, 1] normalized coordinates
    // to the actual pixel coordinates of the window.
    glViewport(0, 0, w, h);

    // We no longer touch glMatrixMode or gluPerspective.
    // The projection matrix will be rebuilt in the display function
    // using the new window dimensions.
}

void renderer_update_camera() {
    // Uncomment this line to rotate the camera around the origin otherwise this function does nothing.
    // g_glData.cameraAngleXY += 0.1f;
    if (g_glData.cameraAngleXY >= 360.0f) g_glData.cameraAngleXY -= 360.0f;
}

void renderer_simulation_and_vbo_update() {
    // Option 1: Run CUDA simulation (preferred if CUDA kernels are ready)
    cuda_run_simulation_step(); // This function will handle interop and ideally update VBO directly or prepare gh_particleSystem.posBuffer


}


// GLUT Callback Implementations
void glut_display_callback() {
    renderer_display_scene();
}

void glut_reshape_callback(GLsizei w, GLsizei h) {
    renderer_handle_resize(w, h);
}

void glut_idle_callback() {
    renderer_simulation_and_vbo_update(); // Runs simulation & VBO update
    renderer_update_camera();             // Updates camera
    glutPostRedisplay();
}

void renderer_release_gl() {
    // ... (Your releaseOpenGL logic, using g_glData.pboID)
    printf("Releasing OpenGL resources...\n");
    if (g_glData.pboID > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &g_glData.pboID);
        printf(" - Deleted VBO ID: %u\n", g_glData.pboID);
        g_glData.pboID = 0;
    }

    if (g_glData.vaoID > 0) {
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &g_glData.vaoID);
        printf(" - Deleted VAO ID: %u\n", g_glData.vaoID);
        g_glData.vaoID = 0;
    }
    if (g_glData.shaderProgramID > 0) {
        glDeleteProgram(g_glData.shaderProgramID);
        printf(" - Deleted Shader Program ID: %u\n", g_glData.shaderProgramID);
        g_glData.shaderProgramID = 0;
    }
    printf("All OpenGL resources released.\n");
}




/**
 * @brief Reads the entire content of a file into a std::string.
 *
 * @param filepath The path to the file.
 * @return The content of the file as a string.
 */
std::string read_file_to_string(const std::string& filepath) {
    std::ifstream file_stream(filepath);
    if (!file_stream.is_open()) {
        std::cerr << "ERROR: Could not open file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file_stream.rdbuf(); // Read the entire file buffer into the stringstream
    return buffer.str();
}


/**
 * @brief Loads shaders from files, compiles them, and links them into a shader program.
 *
 * @param vertex_filepath Path to the vertex shader file.
 * @param fragment_filepath Path to the fragment shader file.
 * @return The ID of the linked shader program, or 0 on failure.
 */
GLuint load_shaders_from_files(const std::string& vertex_filepath, const std::string& fragment_filepath) {
    // 1. Read shader code from files
    std::string vertex_shader_code = read_file_to_string(vertex_filepath);
    std::string fragment_shader_code = read_file_to_string(fragment_filepath);

    if (vertex_shader_code.empty() || fragment_shader_code.empty()) {
        return 0; // Error message was already printed
    }

    // 2. Compile shaders (this part is the same as before, but uses the new strings)
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    GLint Result = GL_FALSE;
    int InfoLogLength;

    // Compile Vertex Shader
    char const* vertex_source_ptr = vertex_shader_code.c_str();
    glShaderSource(VertexShaderID, 1, &vertex_source_ptr, NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader for errors
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        std::cerr << "Vertex Shader Error: " << &VertexShaderErrorMessage[0] << std::endl;
    }

    // Compile Fragment Shader
    char const* fragment_source_ptr = fragment_shader_code.c_str();
    glShaderSource(FragmentShaderID, 1, &fragment_source_ptr, NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader for errors
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        std::cerr << "Fragment Shader Error: " << &FragmentShaderErrorMessage[0] << std::endl;
    }

    // 3. Link the program
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program for errors
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        std::cerr << "Shader Program Link Error: " << &ProgramErrorMessage[0] << std::endl;
    }

    // Shaders are linked, we don't need them anymore
    glDetachShader(ProgramID, VertexShaderID);
    glDetachShader(ProgramID, FragmentShaderID);
    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}
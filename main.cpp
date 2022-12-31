#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Timer.h"
#include "cuda.cuh"

vec3<int> mysize = {100, 100, 16};

void reshapeFunc(int w, int h);

static void checkCudaError(cudaError_t errorCode) {
    if (errorCode != cudaSuccess) {
        fprintf(stderr, "CudaError: %s\n", cudaGetErrorString(errorCode));
    }
}

static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) {
        return;
    }
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_SPACE:
            setFan(!getFan());
            break;
        case GLFW_KEY_P:
            togglePause();
            break;
        case GLFW_KEY_E:
            exportFrame();
            break;
        case GLFW_KEY_I:
            importFrame();
            break;
        default:
            break;
    }
}

static void resize_callback(GLFWwindow* window, int width, int height) {
    reshapeFunc(width, height);
}

void APIENTRY gl_error_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar * message, const void* userParam)
{
    using std::cout;
    using std::endl;
    cout << "---------------------opengl-callback-start------------" << endl;
    cout << "message: " << message << endl;
    cout << "type: ";
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        cout << "ERROR";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        cout << "DEPRECATED_BEHAVIOR";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        cout << "UNDEFINED_BEHAVIOR";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        cout << "PORTABILITY";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        cout << "PERFORMANCE";
        break;
    case GL_DEBUG_TYPE_OTHER:
        cout << "OTHER";
        break;
    }
    cout << endl;

    cout << "id: " << id << endl;
    cout << "severity: ";
    switch (severity) {
    case GL_DEBUG_SEVERITY_LOW:
        cout << "LOW";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        cout << "MEDIUM";
        break;
    case GL_DEBUG_SEVERITY_HIGH:
        cout << "HIGH";
        break;
    }
    cout << endl;
    cout << "---------------------opengl-callback-end--------------" << endl;

}

// Source image on the host side
uchar4* h_Src = nullptr;

// Destination image on the GPU side
uchar4* d_dst = nullptr;

// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource* cuda_pbo_resource;  // handles OpenGL-CUDA exchange

Timer timer;

int imageW = 800, imageH = 600;

GLFWwindow* window;
std::mutex window_mutex;
std::mutex redo_graphics_mutex;
bool dographics = true;

GLuint createShader(const char *code, GLenum type) {
    GLuint shaderid = glCreateShader(type);

    glShaderSource(shaderid, 1, &code, nullptr);
    glCompileShader(shaderid);

    GLint result = GL_FALSE;
    glGetShaderiv(shaderid, GL_COMPILE_STATUS, &result);

    if (result != GL_TRUE) {
        int logLength;
        glGetShaderiv(shaderid, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> errorlog(logLength + 1);
        errorlog[logLength] = 0;
        glGetShaderInfoLog(shaderid, logLength, nullptr, errorlog.data());
        std::cerr << errorlog.data() << "\n";
        return 0;
    }
    return shaderid;
}

GLuint createShaderprogram() {
    GLuint programid = glCreateProgram();
    glAttachShader(programid, createShader(
        "#version 330 core\n"
        "layout(location = 0) in vec2 pos;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  uv = (pos + 1.0) * 0.5;\n"
        "}",
        GL_VERTEX_SHADER)
    );

    glAttachShader(programid, createShader(
        "#version 330 core\n"
        "in vec2 uv;\n"
        "uniform sampler2D image;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "  color = texture(image, uv);\n"
       // "  color = vec4(uv, 0.0, 1.0);\n"
        "}",
        GL_FRAGMENT_SHADER
    ));
    glLinkProgram(programid);
    return programid;
}

void initOpenGLBuffers(int w, int h) {
    // delete old buffers
    if (h_Src) {
        free(h_Src);
        h_Src = nullptr;
    }  

    if (gl_Tex) {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO) {
        checkCudaError(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src = (uchar4*)malloc(w * h * 4);

    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);

    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is only used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it only once.

    checkCudaError(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
}

void reshapeFunc(int w, int h) {
    glViewport(0, 0, w, h);

    
    if (w != 0 && h != 0 && w != imageW && h != imageH) {
        redo_graphics_mutex.lock();
        initOpenGLBuffers(w, h);
        redo_graphics_mutex.unlock();
    }

    imageW = w;
    imageH = h;
    // glutPostRedisplay();
}

void doStuff() {
    setTime(timer.get());

    checkCudaError(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));
    size_t num_bytes;
    checkCudaError(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));

    render(d_dst, imageW, imageH);

    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
}

void graphics() {
    window_mutex.lock();
    glfwMakeContextCurrent(window);
    window_mutex.unlock();
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(gl_error_callback, nullptr);

    long frames = 0;
    glfwSetTime(0);
    timer.start();

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    reshapeFunc(mysize.x, mysize.y);

    redo_graphics_mutex.lock();

    float vertices[] = {
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
     -1.0f,  1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
     -1.0f,  1.0f, 0.0f,
     1.0f, 1.0f, 0.0f
    };

    unsigned int VBO;
    unsigned int VAO;
    unsigned int shader = createShaderprogram();
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    GLint loc_texture = glGetUniformLocation(shader, "image");

    glViewport(0, 0, width, height);

    while (dographics) {

        doStuff();

        glUseProgram(shader);

        glBindTexture(GL_TEXTURE_2D, gl_Tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mysize.x, mysize.y, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, gl_Tex);
        glUniform1i(loc_texture, 0);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        window_mutex.lock();
        if (!dographics) {
            window_mutex.unlock();
            redo_graphics_mutex.unlock();
            break;
        }
        glfwSwapBuffers(window);
        window_mutex.unlock();
        redo_graphics_mutex.unlock();
        std::this_thread::yield();
        redo_graphics_mutex.lock();
    }
    timer.stop();

    printf("%f fps", (float)frames / timer.get());
}

int main() {
    initSimulation(mysize.x, mysize.y, mysize.z);
    togglePause();

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_DEBUG, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    window = glfwCreateWindow(720, 720, "Very good non-game", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);
    // glfwSetWindowSizeCallback(window, resize_callback);

    std::thread graphicsthread = std::thread(graphics);

    while (true) {
        window_mutex.lock();
        if (glfwWindowShouldClose(window)) {
            break;
        }
        window_mutex.unlock();
        glfwPollEvents();
        
    }
    dographics = false;
    window_mutex.unlock();

    graphicsthread.join();

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
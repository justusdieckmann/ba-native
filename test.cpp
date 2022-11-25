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

static void checkCudaError(cudaError_t errorCode) {
    if (errorCode != cudaSuccess) {
        fprintf(stderr, "CudaError: %s\n", cudaGetErrorString(errorCode));
    }
}

static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

int main() {
    initSimulation();

    importFrame();

    // simulateStep();

    printLayer(1);
}
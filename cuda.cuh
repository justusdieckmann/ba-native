#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <iostream>

extern const glm::vec<3, size_t> SIZE;

void render(uchar4* img, int width, int height);

void setTime(float _time);

void initSimulation();

void printLayer(size_t z);

void simulateStep();

void togglePause();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void setFan(bool fan);

bool getFan();

void printP(size_t z);

void exportFrame();

void importFrame();
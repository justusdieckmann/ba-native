#ifndef GASSIMULATION_CUDA_CUH
#define GASSIMULATION_CUDA_CUH

#include <cuda_runtime.h>
#include <iostream>
#include "vec3.h"
#include "Timer.h"

extern Timer timer;
extern double time_split;

void render(uchar4* img, int width, int height);

void initSimulation(size_t xdim, size_t ydim, size_t zdim, size_t gpus, const std::string&);

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

void exportFrame(const std::string& filename);

void importFrame(const std::string&);

#endif //GASSIMULATION_CUDA_CUH
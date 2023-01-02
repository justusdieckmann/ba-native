#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "Timer.h"
#include "cuda.cuh"

static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

int main(int argc, const char** argv) {
    vec3<int> size {100, 100, 100};

    size_t gpus = 1;

    bool doExport = true;

    if (argc > 1) {
        if (argc != 5) {
            std::cerr << "usage: ./test <xdim> <ydim> <zdim> <nGPUs>" << std::endl;
            std::exit(-1);
        }

        size.x = std::atoi(argv[1]);
        size.y = std::atoi(argv[2]);
        size.z = std::atoi(argv[3]);
        gpus = std::atoi(argv[4]);
    }

    initSimulation(size.x, size.y, size.z, gpus);

    Timer t = Timer();

    if (doExport) {
        exportFrame("0.bin");
    }

    t.start();

    for (int i = 0; i < 10; i++) {
        simulateStep();
        std::cout << t.round() << std::endl;
    }

    if (doExport) {
        exportFrame("10.bin");
    }

}
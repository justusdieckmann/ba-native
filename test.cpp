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
    vec3<int> size {200, 200, 200};

    if (argc > 1) {
        if (argc != 4) {
            std::cerr << "usage: ./test <xdim> <ydim> <zdim>" << std::endl;
            std::exit(-1);
        }

        size.x = std::atoi(argv[1]);
        size.y = std::atoi(argv[2]);
        size.z = std::atoi(argv[3]);
    }

    initSimulation(size.x, size.y, size.z);

    Timer t = Timer();

    t.start();

    for (int i = 0; i < 10; i++) {
        simulateStep();
        std::cout << t.round() << std::endl;
    }

}
#include "cuda.cuh"
#include "array.h"
#include <cstdio>
#include <cmath>
#include <fstream>

float pi() { return std::atan(1)*4; }

typedef struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
} floatparts;

const size_t Q = 19;
typedef array<float, Q> cell_t;
typedef vec3<float> vec3f;

const vec3<size_t> SIZE {100, 100, 16};

const size_t CELLS = SIZE.x * SIZE.y * SIZE.z;

__managed__ float deltaT = 0.001f;

__managed__ float tau = 0.0007;
__managed__ float cellwidth = .01f;

__managed__ bool changes = false;

bool fanStatus = false;
bool desiredFanStatus = false;

bool pause = false;

__managed__ float currentTime;

__constant__ const array<vec3f, Q> offsets {
        0, 0, 0,   // 0
        -1, 0, 0,  // 1
        1, 0, 0,   // 2
        0, -1, 0,  // 3
        0, 1, 0,   // 4
        0, 0, -1,  // 5
        0, 0, 1,   // 6
        -1, -1, 0, // 7
        -1, 1, 0,  // 8
        1, -1, 0,  // 9
        1, 1, 0,   // 10
        -1, 0, -1, // 11
        -1, 0, 1,  // 12
        1, 0, -1,  // 13
        1, 0, 1,   // 14
        0, -1, -1, // 15
        0, -1, 1,  // 16
        0, 1, -1,  // 17
        0, 1, 1,   // 18
};

__constant__ const array<unsigned char, Q> opposite = {
        0,
        2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
};

__constant__ const array<float, Q> wis {
    1.f / 3,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
};

cell_t *u1;
cell_t *u2;

cell_t *cudau1;
cell_t *cudau2;

__device__ __host__ int posmod(int a, int b) {
    return (a + b) % b;
}

__device__ __host__ inline size_t pack(size_t w, size_t h, size_t d, size_t x, size_t y, size_t z) {
    return (z * h + y) * w + x;
}

__device__ __host__ inline float feq(size_t i, float p, const vec3f& v) {
    float wi = wis[i];
    float c = cellwidth;
    float dot = offsets[i] * c * v;
    return wi * p * (1 + (1 / (c * c)) * (3 * dot + (9 / (2 * c * c)) * dot * dot - (3.f / 2) * (v * v)));
}

__device__ inline void collisionStep(cell_t &cell) {
    float p = 0;
    float c = cellwidth;
    floatparts* parts = (floatparts*) &cell[0];
    if (parts->exponent == 255) {
        if ((parts->mantissa & 1) != 0) {
            for (size_t i = 1; i < Q; i++) {
                cell[i] = cell[opposite[i]];
            }
        }
        return;
    }
    vec3f vp {0, 0, 0};
    for (size_t i = 0; i < Q; i++) {
        p += cell[i];
        vp += offsets[i] * c * cell[i];
    }
    vec3f v = p == 0 ? vp : vp * (1 / p);

    for (size_t i = 0; i < Q; i++) {
        cell[i] = cell[i] + deltaT / tau * (feq(i, p, v) - cell[i]);
    }
}

__global__ void updateCollision(cell_t *src, vec3<size_t> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= size.x || y >= size.y || z >= size.z) {
        return;
    }
    size_t i = pack(size.x, size.y, size.z, x, y, z);
    collisionStep(src[i]);
}

__global__ void updateStreaming(cell_t *dst, cell_t *src, vec3<size_t> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= size.x || y >= size.y || z >= size.z) {
        return;
    }
    size_t index = pack(size.x, size.y, size.z, x, y, z);

    floatparts* parts = (floatparts*) &src[index][0];

    if (parts->exponent == 255) {
        return;
    }

    for (int i = 1; i < Q; i++) {
        int sx = x + (int) offsets[i].x;
        int sy = y + (int) offsets[i].y;
        int sz = z + (int) offsets[i].z;
        if (sx < 0 || sy < 0 || sz < 0 || sx >= size.x || sy >= size.y || sz >= size.z) {
            continue;
        }
        dst[index][i] = src[pack(size.x, size.y, size.z, sx, sy, sz)][i];
    }
}

__device__ unsigned char floatToChar(float f) {
    return (unsigned char) min(max((f * 100.f + 1.f) * 127.f, 0.f), 255.f);
}

__global__ void renderToBuffer(uchar4 *destImg, cell_t *srcU, vec3<size_t> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 8;

    size_t iP = pack(size.x, size.y, size.z, x, y, z);
    size_t iI = (size.y - y - 1) * size.x + x; // Invert opengl image.
    vec3f p{};
    cell_t cell = srcU[iP];
    for (int i = 0; i < Q; i++) {
        p += offsets[i] * cell[i];
    }
    destImg[iI] = {
            floatToChar(p.x), floatToChar(p.y), floatToChar(p.z), 255
    };

}

void render(uchar4 *img, const int width, const int height) {
    simulateStep();
    dim3 threadsPerBlock(1, 1);
    dim3 numBlocks(SIZE.x, SIZE.y);
    renderToBuffer<<<numBlocks, threadsPerBlock>>>(img, cudau1, SIZE);
    cudaDeviceSynchronize();
}

void setTime(float _time) {
    currentTime = _time;
}

void initSimulation() {
    u1 = new cell_t[CELLS];
    u2 = new cell_t[CELLS];

    gpuErrchk(cudaMalloc(&cudau1, sizeof(cell_t) * CELLS));
    gpuErrchk(cudaMalloc(&cudau2, sizeof(cell_t) * CELLS));
}

void turnOnFan() {
    gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(cell_t) * CELLS, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(u2, cudau2, sizeof(cell_t) * CELLS, cudaMemcpyDeviceToHost));

    int half = SIZE.x / 2;

    for (int x = 0; x < SIZE.x; x++) {
        for (int y = 0; y < SIZE.y; y++) {
            for (int z = 0; z < SIZE.z; z++) {
                float fx = ((int) x - half) / (float) SIZE.x * 2;
                float fy = ((int) y - half) / (float) SIZE.y * 2;
                float angle = atan2(fx, fy) + pi() / 2;
                float d = std::sqrt(fx * fx + fy * fy);
                float strength = std::max(d * (1 - std::pow(10000.f, d - 1)) * 1.f, 0.f);
                vec3f direction = {0, 0, 0};

                for (int i = 0; i < Q; i++) {
                    float f = feq(i, 0.1f, {.001f, 0, 0});
                    u1[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)][i] = f;
                    u2[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)][i] = f;
                }

                if (x <= 1 || y <= 1 || z <= 1 || x >= SIZE.x - 2 || y >= SIZE.y - 2 || z >= SIZE.y - 2 ||  //x == 20 && (y >= 40 && y <= 48 || y >= 52 && y <= 60) || x == 23 && y == 51) {
                    std::pow(x - 50, 2) + std::pow(y - 50, 2) + std::pow(z - 8, 2) <= 225) {
                    floatparts* parts = (floatparts*) &u1[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)][0];
                    parts->sign = 0;
                    parts->exponent = 255;
                    if (x <= 1 || x >= SIZE.x - 2 || y <= 1 || y >= SIZE.y - 2) {
                        parts->mantissa = 1 << 22 | 0b10;
                    } else {
                        parts->mantissa = 1 << 22 | 0b01;
                    }
                    u2[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)][0] = u1[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)][0];
                }
            }
        }
    }

    /*for (size_t i = 0; i < 10; i++) {
        for (size_t z = 0; z < SIZE.z; z++)  {
            size_t index = pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z);
            u1[index][0] = .5f;
            u1[index][1] = .5f;
            u2[index][0] = .5f;
            u2[index][1] = .5f;
        }
    }*/

    gpuErrchk(cudaMemcpy(cudau1, u1, sizeof(cell_t) * CELLS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudau2, u2, sizeof(cell_t) * CELLS, cudaMemcpyHostToDevice));

    printLayer(8);
}

void turnOffFan() {
    gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(cell_t) * CELLS, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(u2, cudau2, sizeof(cell_t) * CELLS, cudaMemcpyDeviceToHost));

    /*for (size_t i = 4; i < 16; i++) {
        for (size_t z = 0; z < SIZE.z; z++)  {
            u1[pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z)] = glm::vec3(0, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z)] = glm::vec3(0, 0, 0);
            u1[pack(SIZE.x, SIZE.y, SIZE.z, i, 0, z)] = glm::vec3(0, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, i, 0, z)] = glm::vec3(0, 0, 0);
            u1[pack(SIZE.x, SIZE.y, SIZE.z, SIZE.x - 1, i, z)] = glm::vec3(0, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, SIZE.x - 1, i, z)] = glm::vec3(0, 0, 0);
            u1[pack(SIZE.x, SIZE.y, SIZE.z, SIZE.x - i, 0, z)] = glm::vec3(0, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, SIZE.x - i, 0, z)] = glm::vec3(0, 0, 0);
        }
    }*/

    gpuErrchk(cudaMemcpy(cudau1, u1, sizeof(cell_t) * CELLS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudau2, u2, sizeof(cell_t) * CELLS, cudaMemcpyHostToDevice));
}

void togglePause() {
    pause = !pause;
}

void setFan(bool fan) {
    desiredFanStatus = fan;
}

bool getFan() {
    return desiredFanStatus;
}

void simulateStep() {
    if (pause) {
        return;
    }
    if (desiredFanStatus != fanStatus) {
        fanStatus = desiredFanStatus;
        if (fanStatus) {
            turnOnFan();
        } else {
            turnOffFan();
        }
    }
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
            (SIZE.x - 2 + threadsPerBlock.x) / threadsPerBlock.x,
            (SIZE.y - 2 + threadsPerBlock.y) / threadsPerBlock.y,
            (SIZE.z - 2 + threadsPerBlock.z) / threadsPerBlock.z
    );
    updateCollision<<<numBlocks, threadsPerBlock>>>(cudau1, SIZE);
    // gpuErrchk(cudaDeviceSynchronize());
    printLayer(8);
    updateStreaming<<<numBlocks, threadsPerBlock>>>(cudau2, cudau1, SIZE);
    std::swap(cudau1, cudau2);
    std::swap(u1, u2);
    gpuErrchk(cudaDeviceSynchronize());
    printLayer(8);
}

void printLayer(size_t z) {
    gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(cell_t) * CELLS, cudaMemcpyDeviceToHost));

    for (size_t y = 0; y < 5u; y++) {
        for (size_t x = 0; x < 5u; x++) {
            cell_t v = u1[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)];
            printf("(%f,%f,%f), ", v[0], v[1], v[2]);
        }
        printf("\n");
    }
    printf("\n");

}

void exportFrame() {
    /*gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(glm::vec3) * CELLS, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(p1, cudap1, sizeof(float) * CELLS, cudaMemcpyDeviceToHost));

    std::ofstream out;
    out.open( "bin.dat", std::ios::out | std::ios::binary);

    for (int x = 1; x < SIZE.x - 1; x++) {
        for (int y = 1; y < SIZE.y - 1; y++) {
            for (int z = 1; z < SIZE.z - 1; z++) {
                int i = pack(SIZE.x, SIZE.y, SIZE.z, x, y, z);
                out.write(reinterpret_cast<const char *>(&u1[i].x), sizeof(float));
                out.write(reinterpret_cast<const char *>(&u1[i].y), sizeof(float));
                out.write(reinterpret_cast<const char *>(&u1[i].z), sizeof(float));
                out.write(reinterpret_cast<const char *>(&p1[i]), sizeof(float));
            }
        }
    }
    out.close();*/
}

void importFrame() {
/*
    turnOffFan();

    std::ifstream in;
    in.open("scenario.dat", std::ios::in | std::ios::binary);

    for (int x = 1; x < SIZE.x - 1; x++) {
        for (int y = 1; y < SIZE.y - 1; y++) {
            for (int z = 1; z < SIZE.z - 1; z++) {
                int i = pack(SIZE.x, SIZE.y, SIZE.z, x, y, z);
                in.read(reinterpret_cast<char *>(&u1[i].x), sizeof(float));
                in.read(reinterpret_cast<char *>(&u1[i].y), sizeof(float));
                in.read(reinterpret_cast<char *>(&u1[i].z), sizeof(float));
                in.read(reinterpret_cast<char *>(&p1[i]), sizeof(float));
            }
        }
    }
    gpuErrchk(cudaMemcpy(cudau1, u1, sizeof(glm::vec3) * CELLS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudap1, p1, sizeof(float) * CELLS, cudaMemcpyHostToDevice));
    in.close();
    */
}


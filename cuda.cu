#include "cuda.cuh"
#include <cstdio>
#include <cmath>

const glm::vec<3, size_t> SIZE(50, 50, 16);

const size_t CELLS = SIZE.x * SIZE.y * SIZE.z;

const float deltaT = 0.005f;

const float viscosity = 0.005;
const float cellwidth = 0.05;

const float EPSILON = 0.0001;

bool fanStatus = false;
bool desiredFanStatus = false;

__managed__ float currentTime;

glm::vec3 *u1;
glm::vec3 *u2;

glm::vec3 *cudau1;
glm::vec3 *cudau2;

float *p1;
float *p2;

float *cudap1;
float *cudap2;

__device__ __host__ inline size_t pack(size_t w, size_t h, size_t d, size_t x, size_t y, size_t z) {
    return (z * h + y) * w + x;
}

__global__ void updateU(glm::vec3 *src, glm::vec3 *dest, float *srcP, glm::vec<3, int> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    size_t i = pack(size.x, size.y, size.z, x, y, z);
    size_t i1n = pack(size.x, size.y, size.z, x - 1, y, z);
    size_t i1p = pack(size.x, size.y, size.z, x + 1, y, z);
    size_t i2n = pack(size.x, size.y, size.z, x, y - 1, z);
    size_t i2p = pack(size.x, size.y, size.z, x, y + 1, z);
    size_t i3n = pack(size.x, size.y, size.z, x, y, z - 1);
    size_t i3p = pack(size.x, size.y, size.z, x, y, z + 1);
    glm::vec3 u = src[i];
    glm::vec3 u1n = src[i1n];
    glm::vec3 u1p = src[i1p];
    glm::vec3 u2n = src[i2n];
    glm::vec3 u2p = src[i2p];
    glm::vec3 u3n = src[i3n];
    glm::vec3 u3p = src[i3p];

    dest[i] = src[i] + deltaT * (viscosity / (cellwidth * cellwidth) * (u1n + u1p + u2n + u2p + u3n + u3p - 6.f * u)
            - 1.f / (2 * cellwidth) * (u1p.x - u1n.x + u2p.y - u2n.y + u3p.z - u3n.z) * u
                    - glm::vec3(srcP[i1n] - srcP[i1p], srcP[i2n] - srcP[i2p], srcP[i3n] - srcP[i3p]) / (2 * cellwidth));
}

__global__ void updatePSingleIteration(glm::vec3 *srcU, const float *srcP, float *destP, glm::vec<3, int> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    size_t i = pack(size.x, size.y, size.z, x, y, z);
    size_t i1n = pack(size.x, size.y, size.z, x - 1, y, z);
    size_t i1p = pack(size.x, size.y, size.z, x + 1, y, z);
    size_t i2n = pack(size.x, size.y, size.z, x, y - 1, z);
    size_t i2p = pack(size.x, size.y, size.z, x, y + 1, z);
    size_t i3n = pack(size.x, size.y, size.z, x, y, z - 1);
    size_t i3p = pack(size.x, size.y, size.z, x, y, z + 1);

    float ud = (cellwidth / 2.f) * (
            srcU[i1p].x - srcU[i1n].x
            + srcU[i2p].y - srcU[i2p].y
            + srcU[i3p].z - srcU[i3p].z
    );

    destP[i] = ((cellwidth * cellwidth) / 6.f) * (srcP[i1p] + srcP[i1n] + srcP[i2p] + srcP[i2n] + srcP[i3p] + srcP[i3n] - ud);
}

__global__ void updateP(glm::vec3 *srcU, float *srcP, glm::vec<3, int> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    size_t i = pack(size.x, size.y, size.z, x, y, z);
    size_t i1n = pack(size.x, size.y, size.z, x - 1, y, z);
    size_t i1p = pack(size.x, size.y, size.z, x + 1, y, z);
    size_t i2n = pack(size.x, size.y, size.z, x, y - 1, z);
    size_t i2p = pack(size.x, size.y, size.z, x, y + 1, z);
    size_t i3n = pack(size.x, size.y, size.z, x, y, z - 1);
    size_t i3p = pack(size.x, size.y, size.z, x, y, z + 1);

    float ud = (cellwidth / 2.f) * (
            srcU[i1p].x - srcU[i1n].x
            + srcU[i2p].y - srcU[i2p].y
            + srcU[i3p].z - srcU[i3p].z
    );

    float oldP = INFINITY;
    float p = srcP[i];
    do {
        oldP = p;
        p = ((cellwidth * cellwidth) / 6.f) * (srcP[i1p] + srcP[i1n] + srcP[i2p] + srcP[i2n] + srcP[i3p] + srcP[i3n] - ud);
        __syncthreads();
        srcP[i] = p;
        __syncthreads();
    } while((abs(p) - abs(oldP)) / (abs(p) + abs(oldP)) > EPSILON);
}

__global__ void updateUFromP(glm::vec3 *destU, const float *srcP, glm::vec<3, int> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    size_t i = pack(size.x, size.y, size.z, x, y, z);
    size_t i1n = pack(size.x, size.y, size.z, x - 1, y, z);
    size_t i1p = pack(size.x, size.y, size.z, x + 1, y, z);
    size_t i2n = pack(size.x, size.y, size.z, x, y - 1, z);
    size_t i2p = pack(size.x, size.y, size.z, x, y + 1, z);
    size_t i3n = pack(size.x, size.y, size.z, x, y, z - 1);
    size_t i3p = pack(size.x, size.y, size.z, x, y, z + 1);

    destU[i] -= glm::vec3(srcP[i1p] - srcP[i1n], srcP[i2p] - srcP[i2n], srcP[i3p] - srcP[i3n]) / (2 * cellwidth);
}

__device__ unsigned char floatToChar(float f) {
    return (unsigned char) min(max((f + 1) * 127.f, 0.f), 255.f);
}

__global__ void renderToBuffer(uchar4 *destImg, glm::vec3 *srcU, glm::vec<3, int> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 8;

    size_t iP = pack(size.x, size.y, size.z, x, y, z);
    size_t iI = y * size.x + x;
    glm::vec3 p = srcU[iP];
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
    u1 = new glm::vec3[CELLS];
    u2 = new glm::vec3[CELLS];

    gpuErrchk(cudaMalloc(&cudau1, sizeof(glm::vec3) * CELLS));
    gpuErrchk(cudaMalloc(&cudau2, sizeof(glm::vec3) * CELLS));

    p1 = new float[CELLS];
    p2 = new float[CELLS];
    gpuErrchk(cudaMalloc(&cudap1, sizeof(float) * CELLS));
    gpuErrchk(cudaMalloc(&cudap2, sizeof(float) * CELLS));
}

void turnOnFan() {
    gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(glm::vec3) * CELLS, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(u2, cudau2, sizeof(glm::vec3) * CELLS, cudaMemcpyDeviceToHost));

    for (size_t i = 6; i < 10; i++) {
        for (size_t z = 0; z < SIZE.z; z++)  {
            u1[pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z)] = glm::vec3(1.f, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z)] = glm::vec3(1.f, 0, 0);
            u1[pack(SIZE.x, SIZE.y, SIZE.z, i, 0, z)] = glm::vec3(0, .75f, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, i, 0, z)] = glm::vec3(0, .75f, 0);
        }
    }

    gpuErrchk(cudaMemcpy(cudau1, u1, sizeof(glm::vec3) * CELLS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudau2, u2, sizeof(glm::vec3) * CELLS, cudaMemcpyHostToDevice));
}

void turnOffFan() {
    gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(glm::vec3) * CELLS, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(u2, cudau2, sizeof(glm::vec3) * CELLS, cudaMemcpyDeviceToHost));

    for (size_t i = 4; i < 12; i++) {
        for (size_t z = 0; z < SIZE.z; z++)  {
            u1[pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z)] = glm::vec3(0, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, 0, i, z)] = glm::vec3(0, 0, 0);
            u1[pack(SIZE.x, SIZE.y, SIZE.z, i, 0, z)] = glm::vec3(0, 0, 0);
            u2[pack(SIZE.x, SIZE.y, SIZE.z, i, 0, z)] = glm::vec3(0, 0, 0);
        }
    }

    gpuErrchk(cudaMemcpy(cudau1, u1, sizeof(glm::vec3) * CELLS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudau2, u2, sizeof(glm::vec3) * CELLS, cudaMemcpyHostToDevice));
}

void setFan(bool fan) {
    desiredFanStatus = fan;
}

bool getFan() {
    return desiredFanStatus;
}

void simulateStep() {
    if (desiredFanStatus != fanStatus) {
        fanStatus = desiredFanStatus;
        if (fanStatus) {
            turnOnFan();
        } else {
            turnOffFan();
        }
    }
    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(SIZE.x - 2, SIZE.y - 2, SIZE.z - 2);
    updateU<<<numBlocks, threadsPerBlock>>>(cudau1, cudau2, cudap1, SIZE);
    gpuErrchk(cudaDeviceSynchronize());
    std::swap(u1, u2);
    std::swap(cudau1, cudau2);
    updateP<<<numBlocks, threadsPerBlock>>>(cudau1, cudap1, SIZE);
    /*for (int i = 0; i < 100; i++) {
        updatePSingleIteration<<<numBlocks, threadsPerBlock>>>(cudau1, cudap1, cudap2, glm::vec3(SIZE, SIZE, SIZE));
        gpuErrchk(cudaDeviceSynchronize());
        std::swap(cudap1, cudap2);
        std::swap(p1, p2);
    }*/
    updateUFromP<<<numBlocks, threadsPerBlock>>>(cudau1, cudap1, SIZE);
    // printLayer(1);
}

void printP(size_t z) {
    gpuErrchk(cudaMemcpy(p1, cudap1, sizeof(float) * CELLS, cudaMemcpyDeviceToHost));
    for (size_t y = 0; y < SIZE.y; y++) {
        for (size_t x = 0; x < SIZE.x; x++) {
            printf("%f, ", p1[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)]);
        }
        printf("\n");
    }
    printf("\n");
}

void printLayer(size_t z) {
    gpuErrchk(cudaMemcpy(u1, cudau1, sizeof(glm::vec3) * CELLS, cudaMemcpyDeviceToHost));

    for (size_t y = 0; y < SIZE.y; y++) {
        for (size_t x = 0; x < SIZE.x; x++) {
            glm::vec3 v = u1[pack(SIZE.x, SIZE.y, SIZE.z, x, y, z)];
            printf("(%f,%f,%f), ", v.x, v.y, v.z);
        }
        printf("\n");
    }
    printf("\n");

}
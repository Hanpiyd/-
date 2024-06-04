#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

constexpr int Dimension = 128;

bool readFvecsFile(const std::string& filename, std::vector<std::vector<float>>& data, int numVecsToRead = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return false;
    }

    int vecCount = 0;
    while (true) {
        if (numVecsToRead != -1 && vecCount >= numVecsToRead) break;

        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof()) break;

        if (dim != Dimension) {
            std::cerr << "Dimension mismatch: expected " << Dimension << ", got " << dim << std::endl;
            return false;
        }

        std::vector<float> vec(Dimension);

        for (int i = 0; i < Dimension; ++i) {
            file.read(reinterpret_cast<char*>(&vec[i]), sizeof(float));
        }

        data.push_back(vec);
        vecCount++;
    }

    file.close();
    return true;
}

__global__ void EuDistanceKernel(const float* __restrict__ dev_data, float* __restrict__ dev_result, int numVecs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVecs) return;

    float sum = 0.0f;
    for (int i = idx + 1; i < numVecs; ++i) {
        float temp_sum = 0.0f;
        for (int j = 0; j < Dimension; ++j) {
            float diff = dev_data[idx * Dimension + j] - dev_data[i * Dimension + j];
            temp_sum += diff * diff;
        }
        sum += sqrtf(temp_sum);
    }
    dev_result[idx] = sum;
}

int main() {
    std::string filename = "sift_base.fvecs";
    std::vector<std::vector<float>> data;

    if (readFvecsFile(filename, data, 200000)) {
        std::cout << "File read successfully." << std::endl;
        std::cout << "Number of vectors read: " << data.size() << std::endl;
    } else {
        std::cerr << "Failed to read file." << std::endl;
        return 1;
    }

    int numVecs = data.size();
    std::vector<float> result(numVecs);

    float* dev_data;
    float* dev_result;

    cudaMalloc(&dev_data, numVecs * Dimension * sizeof(float));
    cudaMalloc(&dev_result, numVecs * sizeof(float));

    cudaMemcpy(dev_data, data.data(), numVecs * Dimension * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    int blockSize = 256;
    int numBlocks = (numVecs + blockSize - 1) / blockSize;

    EuDistanceKernel<<<numBlocks, blockSize>>>(dev_data, dev_result, numVecs);

    cudaMemcpy(result.data(), dev_result, numVecs * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    std::cout << "Time taken: " << duration << " seconds" << std::endl;

    cudaFree(dev_data);
    cudaFree(dev_result);

    return 0;
}

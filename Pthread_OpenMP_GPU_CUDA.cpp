#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

bool readFvecsFile(const string& filename, vector<vector<float>>& data, int numVecsToRead = -1) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Unable to open file: " << filename << endl;
        return false;
    }
    int dimension = 128;
    int int_size = sizeof(int);
    int float_size = sizeof(float);
    int vecCount = 0;
    while (!file.eof() && (numVecsToRead == -1 || vecCount < numVecsToRead)) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), int_size);
        if (file.eof()) break;
        if (dim != dimension) {
            cerr << "Dimension mismatch: expected " << dimension << ", got " << dim << endl;
            return false;
        }
        vector<float> vec(dimension);
        for (int i = 0; i < dimension; ++i) {
            float value;
            file.read(reinterpret_cast<char*>(&value), float_size);
            vec[i] = value;
        }
        data.push_back(vec);
        vecCount++;
    }

    file.close();
    return true;
}

__global__ void euclideanDistanceKernel(const float* a, const float* b, float* distances, int dimension, int numVectors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVectors) {
        float sum = 0;
        for (int i = 0; i < dimension; ++i) {
            float diff = a[idx * dimension + i] - b[idx * dimension + i];
            sum += diff * diff;
        }
        distances[idx] = sqrt(sum);
    }
}

int main() {
    string filename = "sift_base.fvecs";
    vector<vector<float>> data;
    if (readFvecsFile(filename, data, 200000)) {
        cout << "File read successfully." << endl;
        cout << "Number of vectors read: " << data.size() << endl;
    } else {
        cerr << "Failed to read file." << endl;
        return 1;
    }
    int num_vectors = 100000;
    vector<float> h_distances(num_vectors);
    float* d_data;
    cudaMalloc(&d_data, data.size() * data[0].size() * sizeof(float));
    cudaMemcpy(d_data, data.data(), data.size() * data[0].size() * sizeof(float), cudaMemcpyHostToDevice);
    float* d_distances;
    cudaMalloc(&d_distances, num_vectors * sizeof(float));
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_vectors + threadsPerBlock - 1) / threadsPerBlock;
    euclideanDistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, d_distances, data[0].size(), num_vectors);
    cudaMemcpy(h_distances.data(), d_distances, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_distances);
    for (int i = 0; i < 10; ++i) {
        cout << "Distance " << i << ": " << h_distances[i] << endl;
    }

    return 0;
}

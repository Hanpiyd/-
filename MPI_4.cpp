#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <immintrin.h>
#include <xmmintrin.h>
#include <omp.h>
#include <pthread.h>



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

float EuDistance(const std::vector<float>& v1, const std::vector<float>& v2, size_t dim = 128) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string filename = "sift_base.fvecs";
    std::vector<std::vector<float>> data;
    int num_vectors = 200000; // 假设有200000个向量

    if (readFvecsFile(filename, data, num_vectors)) {
        std::cout << "File read successfully." << std::endl;
        std::cout << "Number of vectors read: " << data.size() << std::endl;

        // 划分数据
        int local_data_size = num_vectors / size;
        std::vector<float> local_result(num_vectors, 0.0f);

        // 每个MPI进程内的流水线计算
        for (int i = rank * local_data_size; i < (rank + 1) * local_data_size; ++i) {
            for (int j = 0; j < num_vectors; ++j) {
                local_result[i] += EuDistance(data[i], data[j]);
            }
        }

        // MPI汇总结果
        std::vector<float> global_result(num_vectors, 0.0f);
        MPI_Allreduce(local_result.data(), global_result.data(), num_vectors, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Global result: ";
            for (int i = 0; i < num_vectors; ++i) {
                std::cout << global_result[i] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "Failed to read file." << std::endl;
    }

    MPI_Finalize();
    return 0;
}

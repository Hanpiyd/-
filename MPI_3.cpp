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
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < dim; ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrtf(sum); // 返回欧几里得距离的平方根
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::string filename = "sift_base.fvecs";
    std::vector<std::vector<float>> data;
    if (readFvecsFile(filename, data, 200000)) {
        std::cout << "File read successfully." << std::endl;
        std::cout << "Number of vectors read: " << data.size() << std::endl;
        // 划分数据
        int local_data_size = data.size() / size;
        std::vector<std::vector<float>> local_data(local_data_size);
        MPI_Scatter(data.data(), local_data_size * 128, MPI_FLOAT,
                    local_data.data(), local_data_size * 128, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        // 每个MPI进程内的并行计算
        float local_result = 0.0f;
        #pragma omp parallel for reduction(+:local_result)
        for (int i = 0; i < local_data_size; ++i) {
            for (size_t j = 0; j < data.size(); ++j) {
                local_result += EuDistance(local_data[i], data[j]);
            }
        }
        // MPI汇总结果
        float global_result;
        MPI_Reduce(&local_result, &global_result, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Global result: " << global_result << std::endl;
        }
    } else {
        std::cerr << "Failed to read file." << std::endl;
    }
    MPI_Finalize();
    return 0;
}

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
    for (size_t i = 0; i < dim; i += 8) {
        __m256 v1_load = _mm256_loadu_ps(&v1[i]);
        __m256 v2_load = _mm256_loadu_ps(&v2[i]);
        __m256 diff = _mm256_sub_ps(v1_load, v2_load);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    return sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string filename = "sift_base.fvecs";
    vector<vector<float>> data;

    if (readFvecsFile(filename, data, 200000)) {
        cout << "File read successfully." << endl;
        cout << "Number of vectors read: " << data.size() << endl;

        // 划分数据
        int local_data_size = data.size() / size;
        vector<vector<float>> local_data(local_data_size);

        // 使用单边通信进行数据划分
        MPI_Win win;
        MPI_Win_create(local_data.data(), local_data_size * 128 * sizeof(float), sizeof(float),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);

        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                MPI_Put(data[i].data(), local_data_size * 128, MPI_FLOAT,
                        i, 0, local_data_size * 128, MPI_FLOAT, win);
            }
        }

        MPI_Win_fence(0, win);

        // 使用非阻塞通信进行数据传输
        MPI_Request request;
        MPI_Ibarrier(MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // 每个MPI进程内的并行计算
        float local_result = 0.0f;
#pragma omp parallel for reduction(+:local_result)
        for (int i = 0; i < local_data_size; ++i) {
            local_result += EuDistance(local_data[i], data[i + rank * local_data_size]);
        }

        // MPI汇总结果
        float global_result;
        MPI_Reduce(&local_result, &global_result, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Global result: " << global_result << endl;
        }

        MPI_Win_free(&win);
    } else {
        cerr << "Failed to read file." << endl;
    }

    MPI_Finalize();
    return 0;
}

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
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < dim; i += 8) {
        __m256 v1_load = _mm256_loadu_ps(&v1[i]);
        __m256 v2_load = _mm256_loadu_ps(&v2[i]);
        __m256 diff = _mm256_sub_ps(v1_load, v2_load);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    __m256 perm = _mm256_hadd_ps(sum, sum);
    __m256 shuf = _mm256_permute2f128_ps(perm, perm, 0x1);
    __m256 sums = _mm256_add_ps(perm, shuf);
    __m128 sum_hi128 = _mm256_extractf128_ps(sums, 1);
    __m128 sum_dot128 = _mm256_castps256_ps128(sums);
    sum_dot128 = _mm_add_ps(sum_dot128, sum_hi128);
    __m128 sum_dot = _mm_hadd_ps(sum_dot128, sum_dot128);
    sum_dot = _mm_hadd_ps(sum_dot, sum_dot);
    return sqrtf(_mm_cvtss_f32(sum_dot))/2;
}


int main()
{
    string filename = "sift_base.fvecs";
    vector<vector<float>> data;

    if (readFvecsFile(filename, data, 200000)) {
        cout << "File read successfully." << endl;
        cout << "Number of vectors read: " << data.size() << endl;
        for (float elem : data[10000]) {
                cout << elem << " ";
            }
        cout << endl;
        for (float elem : data[100000]) {
                cout << elem << " ";
            }
        cout << endl;
    } else {
        cerr << "Failed to read file." << endl;
    }
    auto start = chrono::high_resolution_clock::now();
    for(int i=0;i<99999;i++){
        for(int j=i+1;j<100000;j++){
            float temp = EuDistance(data[i],data[j],128);
            if(i==99998){
                cout<<EuDistance(data[i],data[j],128)<<endl;
            }
        }
    }
    auto ed = chrono::high_resolution_clock::now();
    auto duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;
    system("pause");
    return 0;
}

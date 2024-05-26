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

using namespace std;

bool readFvecsFile(const string& filename, vector<vector<float>>& data) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Unable to open file: " << filename << endl;
        return false;
    }


    int dimension = 128;
    int int_size = sizeof(int);
    int float_size = sizeof(float);
    while (!file.eof()) {
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
    }

    file.close();
    return true;
}

//串行算法

float EuDistance(const vector<float>&a,const vector<float>&b,int dimension)
{
    float sum=0;
    for(int i=0;i<dimension;i++){
        float diff=a[i]-b[i];
        sum+=diff*diff;
    }
    sum=sqrt(sum);
    return sum;
}


vector<vector<float>> kmeans(const vector<vector<float>>& data, size_t k,int n) {
    const size_t num_vectors = data.size();
    const size_t vector_dim = data[0].size();
    vector<vector<float>> cluster_centers(k, vector<float>(vector_dim));
    for (size_t i = 0; i < k; ++i) {
        cluster_centers[i] = data[rand() % num_vectors];
    }
    const size_t max_iterations = n;
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        vector<size_t> assignments(num_vectors);
        for (size_t i = 0; i < num_vectors; ++i) {
            float min_distance = numeric_limits<float>::max();
            size_t min_index = 0;
            for (size_t j = 0; j < k; ++j) {
                float distance = EuDistance(data[i], cluster_centers[j],vector_dim);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            assignments[i] = min_index;
        }
        vector<vector<float>> new_cluster_centers(k, vector<float>(vector_dim, 0.0f));
        vector<size_t> cluster_sizes(k, 0);
        for (size_t i = 0; i < num_vectors; ++i) {
            size_t cluster_index = assignments[i];
            for (size_t j = 0; j < vector_dim; ++j) {
                new_cluster_centers[cluster_index][j] += data[i][j];
            }
            cluster_sizes[cluster_index]++;
        }
        for (size_t i = 0; i < k; ++i) {
            if (cluster_sizes[i] > 0) {
                for (size_t j = 0; j < vector_dim; ++j) {
                    new_cluster_centers[i][j] /= static_cast<float>(cluster_sizes[i]);
                }
            } else {
                new_cluster_centers[i] = data[rand() % num_vectors];
            }
        }
        bool converged = true;
        for (size_t i = 0; i < k; ++i) {
            if (EuDistance(new_cluster_centers[i], cluster_centers[i],vector_dim) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) {
            break;
        }
        cout<<"Round: "<<iter<<endl;
        cluster_centers = new_cluster_centers;
    }
    return cluster_centers;
}

vector<vector<float>> generate_cluster_centers_pq(vector<vector<float>>data,size_t num_corewords,size_t num_subspaces)
{
    const size_t vector_dim = data[0].size();
    const size_t subspace_dim = vector_dim / num_subspaces;
    vector<vector<float>> cluster_centers(vector_dim, vector<float>(num_corewords));
    for (size_t i = 0; i < num_subspaces; ++i) {
        vector<vector<float>> subspace_data(data.size(), vector<float>(subspace_dim));
        for (size_t j = 0; j < data.size(); ++j) {
            for (size_t k = 0; k < subspace_dim; ++k) {
                subspace_data[j][k] = data[j][i * subspace_dim + k];
            }
        }
        vector<vector<float>> subspace_cluster_centers = kmeans(subspace_data, num_corewords, 2);
        for (size_t j = 0; j < num_corewords; ++j) {
            for (size_t k = 0; k < subspace_dim; ++k) {
                cluster_centers[i * subspace_dim + k][j] = subspace_cluster_centers[j][k];
            }
        }
    }
    return cluster_centers;
}


//AVX并行


float EuDistanceAVX(const std::vector<float>& v1, const std::vector<float>& v2, size_t dim = 128) {
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

vector<vector<float>> kmeansAVX(const vector<vector<float>>& data, size_t k, int n) {
    const size_t num_vectors = data.size();
    const size_t vector_dim = data[0].size();
    vector<vector<float>> cluster_centers(k, vector<float>(vector_dim));
    for (size_t i = 0; i < k; i++) {
        cluster_centers[i] = data[rand() % num_vectors];
    }
    const size_t max_iterations = n;
    for (size_t iter = 0; iter < max_iterations; iter++) {
        vector<size_t> assignments(num_vectors);
        for (size_t i = 0; i < num_vectors; ++i) {
            float min_distance = std::numeric_limits<float>::max();
            size_t min_index = 0;
            for (size_t j = 0; j < k; ++j) {
                float distance = EuDistanceAVX(data[i], cluster_centers[j], vector_dim);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            assignments[i] = min_index;
        }
        vector<vector<float>> new_cluster_centers(k, vector<float>(vector_dim, 0.0f));
        vector<size_t> cluster_sizes(k, 0);
        for (size_t i = 0; i < num_vectors; ++i) {
            const float* data_ptr = data[i].data();
            size_t cluster_index = assignments[i];
            float* center_ptr = new_cluster_centers[cluster_index].data();
            for (size_t j = 0; j < vector_dim; j += 8) {
                __m256 vec_data = _mm256_loadu_ps(data_ptr + j);
                __m256 vec_center = _mm256_loadu_ps(center_ptr + j);
                __m256 vec_result = _mm256_add_ps(vec_data, vec_center);
                _mm256_storeu_ps(center_ptr + j, vec_result);
            }
            cluster_sizes[cluster_index]++;
        }
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < vector_dim; ++j) {
                if (cluster_sizes[i] > 0) {
                        cout<<cluster_sizes[i]<<endl;
                    new_cluster_centers[i][j] /= static_cast<float>(cluster_sizes[i]);
                } else {
                    new_cluster_centers[i] = data[rand() % num_vectors];
                }
            }
        }
        bool converged = true;
        for (size_t i = 0; i < k; ++i) {
            if (EuDistanceAVX(new_cluster_centers[i], cluster_centers[i], vector_dim) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) {
            break;
        }
        cluster_centers = new_cluster_centers;
        cout << "Round: " << iter << endl;
    }
    return cluster_centers;
}

vector<vector<float>> generate_cluster_centers_pqAVX(vector<vector<float>> data, size_t num_corewords, size_t num_subspaces) {
    const size_t vector_dim = data[0].size();
    const size_t subspace_dim = vector_dim / num_subspaces;
    vector<vector<float>> cluster_centers(vector_dim, vector<float>(num_corewords));
    for (size_t i = 0; i < num_subspaces; ++i) {
        vector<vector<float>> subspace_data(data.size(), vector<float>(subspace_dim));
        for (size_t j = 0; j < data.size(); ++j) {
            for (size_t k = 0; k < subspace_dim; ++k) {
                subspace_data[j][k] = data[j][i * subspace_dim + k];
            }
        }
        vector<vector<float>> subspace_cluster_centers = kmeansAVX(subspace_data, num_corewords, 2);
        for (size_t j = 0; j < num_corewords; ++j) {
            float* center_ptr = cluster_centers[i * subspace_dim + j].data();
            for (size_t k = 0; k < subspace_dim; k += 8) {
                __m256 vec_center = _mm256_loadu_ps(center_ptr + k);
                __m256 vec_subspace_center = _mm256_loadu_ps(subspace_cluster_centers[j].data() + k);
                _mm256_storeu_ps(center_ptr + k, vec_subspace_center);
            }
        }
    }

    return cluster_centers;
}

//SSE并行

float EuDistanceSSE(const vector<float>& a, const vector<float>& b, int dimension) {
    float sum = 0.0f;
    int numFloats = dimension / 4;
    __m128 diffSum = _mm_setzero_ps();
    for (int i = 0; i < numFloats; ++i) {
        __m128 va = _mm_loadu_ps(&a[i * 4]);
        __m128 vb = _mm_loadu_ps(&b[i * 4]);
        __m128 diff = _mm_sub_ps(va, vb);
        diff = _mm_mul_ps(diff, diff);
        diffSum = _mm_add_ps(diffSum, diff);
    }
    float temp[4];
    _mm_storeu_ps(temp, diffSum);
    for (int i = 0; i < 4; ++i) {
        sum += temp[i];
    }
    for (int i = numFloats * 4; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

vector<vector<float>> kmeansSSE(const vector<vector<float>>& data, size_t k, int n) {
    const size_t num_vectors = data.size();
    const size_t vector_dim = data[0].size();
    vector<vector<float>> cluster_centers(k, vector<float>(vector_dim));
    for (size_t i = 0; i < k; i++) {
        cluster_centers[i] = data[rand() % num_vectors];
    }
    const size_t max_iterations = n;
    for (size_t iter = 0; iter < max_iterations; iter++) {
        vector<size_t> assignments(num_vectors);
        for (size_t i = 0; i < num_vectors; ++i) {
            float min_distance = std::numeric_limits<float>::max();
            size_t min_index = 0;

            for (size_t j = 0; j < k; ++j) {
                float distance = EuDistanceSSE(data[i], cluster_centers[j], vector_dim);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }

            assignments[i] = min_index;
        }
        vector<vector<float>> new_cluster_centers(k, vector<float>(vector_dim, 0.0f));
        vector<size_t> cluster_sizes(k, 0);
        for (size_t i = 0; i < num_vectors; ++i) {
            const float* data_ptr = data[i].data();
            size_t cluster_index = assignments[i];
            float* center_ptr = new_cluster_centers[cluster_index].data();
            for (size_t j = 0; j < vector_dim; j += 4) {
                __m128 vec_data = _mm_loadu_ps(data_ptr + j);
                __m128 vec_center = _mm_loadu_ps(center_ptr + j);
                __m128 vec_result = _mm_add_ps(vec_data, vec_center);
                _mm_storeu_ps(center_ptr + j, vec_result);
            }
            cluster_sizes[cluster_index]++;
        }
        for (size_t i = 0; i < k; ++i) {
            if (cluster_sizes[i] > 0) {
                for (size_t j = 0; j < vector_dim; ++j) {
                    new_cluster_centers[i][j] /= static_cast<float>(cluster_sizes[i]);
                }
            } else {
                new_cluster_centers[i] = data[rand() % num_vectors];
            }
        }
        bool converged = true;
        for (size_t i = 0; i < k; ++i) {
            if (EuDistanceSSE(new_cluster_centers[i], cluster_centers[i], vector_dim) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) {
            break;
        }
        cluster_centers = new_cluster_centers;
        cout << "Round: " << iter << endl;
    }
    return cluster_centers;
}

vector<vector<float>> generate_cluster_centers_pq_SSE(const vector<vector<float>>& data, size_t num_corewords, size_t num_subspaces) {
    const size_t vector_dim = data[0].size();
    const size_t subspace_dim = vector_dim / num_subspaces;
    vector<vector<float>> cluster_centers(vector_dim, vector<float>(num_corewords));

    for (size_t i = 0; i < num_subspaces; ++i) {
        vector<vector<float>> subspace_data(data.size(), vector<float>(subspace_dim));
        for (size_t j = 0; j < data.size(); ++j) {
            for (size_t k = 0; k < subspace_dim; ++k) {
                subspace_data[j][k] = data[j][i * subspace_dim + k];
            }
        }
        vector<vector<float>> subspace_cluster_centers = kmeansSSE(subspace_data, num_corewords, 2);
        for (size_t j = 0; j < num_corewords; ++j) {
            float* center_ptr = cluster_centers[i * subspace_dim + j].data();
            for (size_t k = 0; k < subspace_dim; k += 4) {
                __m128 vec_center = _mm_loadu_ps(center_ptr + k);
                __m128 vec_subspace_center = _mm_loadu_ps(subspace_cluster_centers[j].data() + k);
                _mm_storeu_ps(center_ptr + k, vec_subspace_center);
            }
        }
    }
    return cluster_centers;
}



int main()
{
    string filename = "sift_base.fvecs";
    vector<vector<float>> data;

    if (readFvecsFile(filename, data)) {
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
    for(int i=0;i<29999;i++){
        for(int j=i+1;j<30000;j++){
            float temp = EuDistance(data[i],data[j],128);
            if(i==29998){
                cout<<EuDistance(data[i],data[j],128)<<endl;
            }
        }
    }
    auto ed = chrono::high_resolution_clock::now();
    auto duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;

    start = chrono::high_resolution_clock::now();
    for(int i=0;i<29999;i++){
        for(int j=i+1;j<30000;j++){
            float temp = EuDistanceAVX(data[i],data[j],128);
            if(i==29998){
                cout<<EuDistanceAVX(data[i],data[j],128)<<endl;
            }
        }
    }
    ed = chrono::high_resolution_clock::now();
    duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;

    start = chrono::high_resolution_clock::now();
    for(int i=0;i<29999;i++){
        for(int j=i+1;j<30000;j++){
            float temp = EuDistanceSSE(data[i],data[j],128);
            if(i==29998){
                cout<<EuDistanceSSE(data[i],data[j],128)<<endl;
            }
        }
    }
    ed = chrono::high_resolution_clock::now();
    duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;
//    start = chrono::high_resolution_clock::now();
//    vector<vector<float>> cluster_centers = generate_cluster_centers_pq(data, 256, 8);
//    ed = chrono::high_resolution_clock::now();
//    duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
//    cout << "Time taken: " << duration << " seconds" << endl;
//    cout << "Cluster centers:" << endl;
//    for (size_t i = 0; i < 256; ++i) {
//        cout << "Center " << i << ": ";
//        for (size_t j = 0; j < 128; ++j) {
//            cout << cluster_centers[j][i] << " ";
//        }
//        cout <<endl;
//    }
//    cout<<endl;

    system("pause");
    return 0;
}

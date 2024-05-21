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

struct ThreadData {
    int begin;
    int finish;
    const vector<vector<float>>* data;
    vector<vector<float>>* results;
    int dimension;
    pthread_mutex_t* mutex;
};

void* computeDistances(void* arg) {
    ThreadData* td = reinterpret_cast<ThreadData*>(arg);
    const vector<vector<float>>& data = *(td->data);
    vector<vector<float>>& results = *(td->results);
    int dimension = td->dimension;

    for (int i = td->begin; i < td->finish; ++i) {
        (*td->results)[i].resize(100000);
        for (int j = 0; j < 100000; ++j) {
            float distance = EuDistance(data[i], data[j], dimension);
            //pthread_mutex_lock(td->mutex);
            (*td->results)[i][j] = distance;
            //pthread_mutex_unlock(td->mutex);
        }
    }

    pthread_exit(nullptr);
}

vector<vector<float>> kmeans(const vector<vector<float>>& data, size_t k, int n) {
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
                float distance = EuDistance(data[i], cluster_centers[j], vector_dim);
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
            if (EuDistance(new_cluster_centers[i], cluster_centers[i], vector_dim) > 1e-6) {
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

struct ThreadData_1 {
    const vector<vector<float>>& data;
    const vector<vector<float>>& cluster_centers;
    vector<size_t>& assignments;
    const size_t begin;
    const size_t finish;
    const size_t k;
    const size_t vector_dim;
};

void* calculateAssignments(void* thread_arg) {
    ThreadData_1* data = reinterpret_cast<ThreadData_1*>(thread_arg);
    const vector<vector<float>>& data_ref = data->data;
    const vector<vector<float>>& centers_ref = data->cluster_centers;
    vector<size_t>& assignments_ref = data->assignments;
    const size_t begin = data->begin;
    const size_t finish = data->finish;
    const size_t k = data->k;
    const size_t vector_dim = data->vector_dim;
    for (size_t i = begin; i < finish; ++i) {
        float min_distance = numeric_limits<float>::max();
        size_t min_index = 0;
        for (size_t j = 0; j < k; ++j) {
            float distance = EuDistanceAVX(data_ref[i], centers_ref[j], vector_dim);
            if (distance < min_distance) {
                min_distance = distance;
                min_index = j;
            }
        }
        assignments_ref[i] = min_index;
    }
    pthread_exit(NULL);
}


vector<vector<float>> kmeansAVXPthread(const vector<vector<float>>& data, size_t k, int n, int num_threads) {
    const size_t num_vectors = data.size();
    const size_t vector_dim = data[0].size();
    vector<vector<float>> cluster_centers(k, vector<float>(vector_dim));
    for (size_t i = 0; i < k; i++) {
        cluster_centers[i] = data[rand() % num_vectors];
    }
    const size_t max_iterations = n;
    for (size_t iter = 0; iter < max_iterations; iter++) {
        vector<size_t> assignments(num_vectors);
        vector<ThreadData_1> thread_data(num_threads);
        size_t chunk_size = num_vectors / num_threads;
        size_t remainder = num_vectors % num_threads;
        size_t begin = 0;
        for (int i = 0; i < num_threads; ++i) {
            size_t finish = begin + chunk_size + (i < remainder ? 1 : 0);
            thread_data[i] = {data, cluster_centers, assignments, begin, finish, k, vector_dim};
            begin = finish;
        }
        pthread_t threads[num_threads];
        for (int i = 0; i < num_threads; ++i) {
            int rc = pthread_create(&threads[i], NULL, calculateAssignments, reinterpret_cast<void*>(&thread_data[i]));
            if (rc) {
                cerr << "Error: Unable to create thread, " << rc << endl;
                exit(-1);
            }
        }
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
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

vector<vector<float>> kmeansOpenMP(const vector<vector<float>>& data, size_t k, int n) {
    const size_t num_vectors = data.size();
    const size_t vector_dim = data[0].size();
    vector<vector<float>> cluster_centers(k, vector<float>(vector_dim));
    #pragma omp parallel for
    for (size_t i = 0; i < k; i++) {
        cluster_centers[i] = data[rand() % num_vectors];
    }
    const size_t max_iterations = n;
    for (size_t iter = 0; iter < max_iterations; iter++) {
        vector<size_t> assignments(num_vectors);
        #pragma omp parallel for
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
        #pragma omp parallel for
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
            #pragma omp atomic
            cluster_sizes[cluster_index]++;
        }
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < vector_dim; ++j) {
                if (cluster_sizes[i] > 0) {
                    new_cluster_centers[i][j] /= static_cast<float>(cluster_sizes[i]);
                } else {
                    new_cluster_centers[i] = data[rand() % num_vectors];
                }
            }
        }
        bool converged = true;
        #pragma omp parallel for shared(converged)
        for (size_t i = 0; i < k; ++i) {
            if (EuDistanceAVX(new_cluster_centers[i], cluster_centers[i], vector_dim) > 1e-6) {
                #pragma omp atomic write
                converged = false;
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


    int num_vectors = 100000;
    vector<vector<float>> distances(num_vectors, vector<float>(num_vectors));
    #pragma omp parallel for schedule(dynamic)
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = i + 1; j < num_vectors; ++j) {
            float distance = EuDistance(data[i], data[j], data[i].size());
            #pragma omp atomic write
            distances[i][j] = distance;
        }
    }
    ed = chrono::high_resolution_clock::now();
    duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;

    vector<vector<float>> results(100000);
    int num_threads = 4;
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    int vectors_per_thread = 100000 / num_threads;
    int remaining_vectors = 100000 % num_threads;
    int begin = 0;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        td[i].begin = begin;
        td[i].finish = begin + vectors_per_thread + (i < remaining_vectors ? 1 : 0);
        td[i].data = &data;
        td[i].results = &results;
        td[i].dimension = data[0].size();
        pthread_create(&threads[i], nullptr, computeDistances, &td[i]);
        begin = td[i].finish;
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    ed = chrono::high_resolution_clock::now();
    duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;
    system("pause");
    return 0;
}

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace std;
using namespace oneapi::dpl::execution;

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

float EuDistance(const vector<float>& a, const vector<float>& b, int dimension) {
    float sum = 0;
    __m128 diff_sum = _mm_setzero_ps();
    for (int i = 0; i < dimension; i += 4) {
        __m128 vec_a = _mm_loadu_ps(&a[i]);
        __m128 vec_b = _mm_loadu_ps(&b[i]);
        __m128 diff = _mm_sub_ps(vec_a, vec_b);
        diff = _mm_mul_ps(diff, diff);
        diff_sum = _mm_add_ps(diff_sum, diff);
    }
    float temp[4];
    _mm_storeu_ps(temp, diff_sum);
    for (int i = 0; i < 4; ++i) {
        sum += temp[i];
    }
    for (int i = dimension - dimension % 4; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
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
        (*td->results)[i].resize(data.size());
        for (int j = 0; j < data.size(); ++j) {
            float distance = EuDistance(data[i], data[j], dimension);
            (*td->results)[i][j] = distance;
        }
    }
    pthread_exit(nullptr);
}

int main() {
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
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 99999; i++) {
        for (int j = i + 1; j < 100000; j++) {
            float temp = EuDistance(data[i], data[j], 128);
            if (i == 99998) {
                cout << EuDistance(data[i], data[j], 128) << endl;
            }
        }
    }
    auto ed = chrono::high_resolution_clock::now();
    auto duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;

    int num_vectors = 100000;
    vector<vector<float>> distances(num_vectors, vector<float>(num_vectors));
    auto exec = par_unseq;
    start = chrono::high_resolution_clock::now();
    oneapi::dpl::for_each(exec, oneapi::dpl::counting_iterator<int>(0), oneapi::dpl::counting_iterator<int>(num_vectors),
        [=](int i) {
            for (int j = i + 1; j < num_vectors; ++j) {
                float distance = EuDistance(data[i], data[j], data[i].size());
                distances[i][j] = distance;
            }
        });
    ed = chrono::high_resolution_clock::now();
    duration = (float)(chrono::duration_cast<chrono::nanoseconds>(ed - start).count() / 1.0e9);
    cout << "Time taken: " << duration << " seconds" << endl;
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = i + 1; j < num_vectors; ++j) {
            cout << "Distance between vector " << i << " and vector " << j << ": " << distances[i][j] << endl;
        }
    }

    return 0;
}

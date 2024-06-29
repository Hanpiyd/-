#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cfloat>
#include <random>
#include <immintrin.h> // SIMDͷ�ļ�
#include <omp.h>       // OpenMPͷ�ļ�

// ���峣��
const int DIM = 128;        // ������ά��
const int BASE_VECS = 200000;  // ��ȡ�Ļ���������
const int QUERY_VECS = 10000;  // ��ѯ��������
const int K = 16;           // ������������

// ����ṹ���ʾ����
struct Vector {
    float data[DIM];
};

// ��ȡfvecs�ļ�����
void read_fvecs(const char* filename, std::vector<Vector>& vecs, int max_vecs) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "�޷����ļ�: " << filename << std::endl;
        exit(1);
    }
    int dim;
    while (file.read((char*)&dim, sizeof(int)) && vecs.size() < max_vecs) {
        Vector vec;
        file.read((char*)vec.data, sizeof(float) * DIM);
        vecs.push_back(vec);
    }
}

// ����ŷ�Ͼ��뺯����SIMD�汾��
float simd_euclidean_distance(const Vector& a, const Vector& b) {
    float dist = 0;
    __m256 diff, square;

    for (int i = 0; i < DIM; i += 8) {
        // Load vectors into SIMD registers
        __m256 va = _mm256_loadu_ps(a.data + i);
        __m256 vb = _mm256_loadu_ps(b.data + i);

        // Compute squared differences
        diff = _mm256_sub_ps(va, vb);
        square = _mm256_mul_ps(diff, diff);

        // Accumulate to dist
        dist += _mm256_reduce_add_ps(square);
    }

    // Compute square root
    return std::sqrt(dist);
}

// k-means�����㷨��OpenMP���л���
void kmeans(const std::vector<Vector>& vecs, std::vector<Vector>& centroids, std::vector<int>& assignments, int k, int max_iters = 100) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, vecs.size() - 1);

    // �����ʼ����������
    for (int i = 0; i < k; ++i) {
        centroids.push_back(vecs[dis(gen)]);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // ����ÿ�������������������
        #pragma omp parallel for
        for (size_t i = 0; i < vecs.size(); ++i) {
            float min_dist = FLT_MAX;
            int best_cluster = -1;
            for (int j = 0; j < k; ++j) {
                float dist = simd_euclidean_distance(vecs[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // ���¾�������
        std::vector<Vector> new_centroids(k);
        std::vector<int> counts(k, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < vecs.size(); ++i) {
            int cluster = assignments[i];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp atomic
                new_centroids[cluster].data[d] += vecs[i].data[d];
            }
            #pragma omp atomic
            counts[cluster]++;
        }
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int d = 0; d < DIM; ++d) {
                    new_centroids[j].data[d] /= counts[j];
                }
            } else {
                // ���һ����������û���κγ�Ա��������³�ʼ��
                new_centroids[j] = vecs[dis(gen)];
            }
        }
        centroids = new_centroids;
    }
}

// �в�����������SIMD�汾��
void residual_quantization(const std::vector<Vector>& vecs, const std::vector<Vector>& centroids, std::vector<Vector>& residuals, const std::vector<int>& assignments) {
    residuals.resize(vecs.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vecs.size(); ++i) {
        int cluster = assignments[i];
        for (int j = 0; j < DIM; ++j) {
            residuals[i].data[j] = vecs[i].data[j] - centroids[cluster].data[j];
        }
    }
}

// �������������������OpenMP���л���
std::vector<int> approximate_nearest_neighbors(const std::vector<Vector>& queries, const std::vector<Vector>& base, const std::vector<Vector>& residuals, const std::vector<Vector>& centroids, const std::vector<int>& assignments) {
    std::vector<int> results(queries.size(), -1);
    #pragma omp parallel for
    for (size_t q = 0; q < queries.size(); ++q) {
        float min_dist = FLT_MAX;
        int best_index = -1;
        for (size_t i = 0; i < base.size(); ++i) {
            // �����ѯ�������������ĵľ���
            float dist_to_centroid = simd_euclidean_distance(queries[q], centroids[assignments[i]]);
            // ���ϲв�ľ���
            float dist = dist_to_centroid + simd_euclidean_distance(queries[q], residuals[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_index = i;
            }
        }
        results[q] = best_index;
    }
    return results;
}

int main() {
    // ��ȡ�������Ͳ�ѯ����
    std::vector<Vector> base_vecs;
    read_fvecs("sift-base.fvecs", base_vecs, BASE_VECS);
    std::vector<Vector> query_vecs;
    read_fvecs("sift-query.fvecs", query_vecs, QUERY_VECS);

    // ����
    std::vector<Vector> centroids;
    std::vector<int> assignments(base_vecs.size());
    kmeans(base_vecs, centroids, assignments, K);

    // ����в�
    std::vector<Vector> residuals;
    residual_quantization(base_vecs, centroids, residuals, assignments);

    // �������������
    std::vector<int> results = approximate_nearest_neighbors(query_vecs, base_vecs, residuals, centroids, assignments);

    // ������
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "��ѯ���� " << i << " �������������: " << results[i] << std::endl;
    }

    return 0;
}

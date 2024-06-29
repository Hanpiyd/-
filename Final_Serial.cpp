#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cfloat>
#include <random>

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

// ����ŷ�Ͼ��뺯��
float euclidean_distance(const Vector& a, const Vector& b) {
    float dist = 0;
    for (int i = 0; i < DIM; ++i) {
        float diff = a.data[i] - b.data[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// k-means�����㷨
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
        for (size_t i = 0; i < vecs.size(); ++i) {
            float min_dist = FLT_MAX;
            int best_cluster = -1;
            for (int j = 0; j < k; ++j) {
                float dist = euclidean_distance(vecs[i], centroids[j]);
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
        for (size_t i = 0; i < vecs.size(); ++i) {
            int cluster = assignments[i];
            for (int j = 0; j < DIM; ++j) {
                new_centroids[cluster].data[j] += vecs[i].data[j];
            }
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

// �в���������
void residual_quantization(const std::vector<Vector>& vecs, const std::vector<Vector>& centroids, std::vector<Vector>& residuals, const std::vector<int>& assignments) {
    for (size_t i = 0; i < vecs.size(); ++i) {
        Vector residual;
        int cluster = assignments[i];
        for (int j = 0; j < DIM; ++j) {
            residual.data[j] = vecs[i].data[j] - centroids[cluster].data[j];
        }
        residuals.push_back(residual);
    }
}

// �����������������
std::vector<int> approximate_nearest_neighbors(const std::vector<Vector>& queries, const std::vector<Vector>& base, const std::vector<Vector>& residuals, const std::vector<Vector>& centroids, const std::vector<int>& assignments) {
    std::vector<int> results;
    for (const auto& query : queries) {
        float min_dist = FLT_MAX;
        int best_index = -1;
        for (size_t i = 0; i < base.size(); ++i) {
            // �����ѯ�������������ĵľ���
            float dist_to_centroid = euclidean_distance(query, centroids[assignments[i]]);
            // ���ϲв�ľ���
            float dist = dist_to_centroid + euclidean_distance(query, residuals[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_index = i;
            }
        }
        results.push_back(best_index);
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
    for (int result : results) {
        std::cout << "���������: " << result << std::endl;
    }

    return 0;
}

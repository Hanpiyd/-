#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cfloat>
#include <random>
#include <immintrin.h> // SIMD头文件
#include <omp.h>       // OpenMP头文件

// 定义常量
const int DIM = 128;        // 向量的维度
const int BASE_VECS = 200000;  // 读取的基向量数量
const int QUERY_VECS = 10000;  // 查询向量数量
const int K = 16;           // 聚类中心数量

// 定义结构体表示向量
struct Vector {
    float data[DIM];
};

// 读取fvecs文件函数
void read_fvecs(const char* filename, std::vector<Vector>& vecs, int max_vecs) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    int dim;
    while (file.read((char*)&dim, sizeof(int)) && vecs.size() < max_vecs) {
        Vector vec;
        file.read((char*)vec.data, sizeof(float) * DIM);
        vecs.push_back(vec);
    }
}

// 计算欧氏距离函数（SIMD版本）
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

// k-means聚类算法（OpenMP并行化）
void kmeans(const std::vector<Vector>& vecs, std::vector<Vector>& centroids, std::vector<int>& assignments, int k, int max_iters = 100) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, vecs.size() - 1);

    // 随机初始化聚类中心
    for (int i = 0; i < k; ++i) {
        centroids.push_back(vecs[dis(gen)]);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // 计算每个向量的最近聚类中心
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

        // 更新聚类中心
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
                // 如果一个聚类中心没有任何成员，随机重新初始化
                new_centroids[j] = vecs[dis(gen)];
            }
        }
        centroids = new_centroids;
    }
}

// 残差量化函数（SIMD版本）
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

// 近似最近邻搜索函数（OpenMP并行化）
std::vector<int> approximate_nearest_neighbors(const std::vector<Vector>& queries, const std::vector<Vector>& base, const std::vector<Vector>& residuals, const std::vector<Vector>& centroids, const std::vector<int>& assignments) {
    std::vector<int> results(queries.size(), -1);
    #pragma omp parallel for
    for (size_t q = 0; q < queries.size(); ++q) {
        float min_dist = FLT_MAX;
        int best_index = -1;
        for (size_t i = 0; i < base.size(); ++i) {
            // 计算查询向量到聚类中心的距离
            float dist_to_centroid = simd_euclidean_distance(queries[q], centroids[assignments[i]]);
            // 加上残差的距离
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
    // 读取基向量和查询向量
    std::vector<Vector> base_vecs;
    read_fvecs("sift-base.fvecs", base_vecs, BASE_VECS);
    std::vector<Vector> query_vecs;
    read_fvecs("sift-query.fvecs", query_vecs, QUERY_VECS);

    // 聚类
    std::vector<Vector> centroids;
    std::vector<int> assignments(base_vecs.size());
    kmeans(base_vecs, centroids, assignments, K);

    // 计算残差
    std::vector<Vector> residuals;
    residual_quantization(base_vecs, centroids, residuals, assignments);

    // 近似最近邻搜索
    std::vector<int> results = approximate_nearest_neighbors(query_vecs, base_vecs, residuals, centroids, assignments);

    // 输出结果
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "查询向量 " << i << " 的最近邻索引是: " << results[i] << std::endl;
    }

    return 0;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <cuda_runtime.h>

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

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err \
                      << " \"" << cudaGetErrorString(err) << "\" \n"; \
            exit(1); \
        } \
    } while (0)

// k-means聚类算法的CUDA核函数
__global__ void kmeans_kernel(const Vector* vecs, Vector* centroids, int* assignments, int num_vecs, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vecs) return;

    float min_dist = FLT_MAX;
    int best_cluster = -1;
    for (int j = 0; j < k; ++j) {
        float dist = 0;
        for (int d = 0; d < dim; ++d) {
            float diff = vecs[idx].data[d] - centroids[j].data[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = j;
        }
    }
    assignments[idx] = best_cluster;
}

// k-means聚类算法
void kmeans(const std::vector<Vector>& vecs, std::vector<Vector>& centroids, std::vector<int>& assignments, int k, int max_iters = 100) {
    int num_vecs = vecs.size();
    int dim = DIM;

    Vector* d_vecs;
    Vector* d_centroids;
    int* d_assignments;

    // 分配CUDA内存
    CUDA_CHECK(cudaMalloc((void**)&d_vecs, num_vecs * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, k * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, num_vecs * sizeof(int)));

    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_vecs, vecs.data(), num_vecs * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Vector), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_vecs + blockSize - 1) / blockSize;

    for (int iter = 0; iter < max_iters; ++iter) {
        kmeans_kernel<<<numBlocks, blockSize>>>(d_vecs, d_centroids, d_assignments, num_vecs, k, dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 复制分配结果回主机
        CUDA_CHECK(cudaMemcpy(assignments.data(), d_assignments, num_vecs * sizeof(int), cudaMemcpyDeviceToHost));

        // 更新聚类中心
        std::vector<Vector> new_centroids(k);
        std::vector<int> counts(k, 0);
        for (size_t i = 0; i < vecs.size(); ++i) {
            int cluster = assignments[i];
            for (int j = 0; j < dim; ++j) {
                new_centroids[cluster].data[j] += vecs[i].data[j];
            }
            counts[cluster]++;
        }
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int d = 0; d < dim; ++d) {
                    new_centroids[j].data[d] /= counts[j];
                }
            } else {
                // 如果一个聚类中心没有任何成员，随机重新初始化
                new_centroids[j] = vecs[rand() % vecs.size()];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_centroids, new_centroids.data(), k * sizeof(Vector), cudaMemcpyHostToDevice));
    }

    // 复制聚类中心回主机
    CUDA_CHECK(cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Vector), cudaMemcpyDeviceToHost));

    // 释放CUDA内存
    CUDA_CHECK(cudaFree(d_vecs));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
}

// 残差量化的CUDA核函数
__global__ void residual_quantization_kernel(const Vector* vecs, const Vector* centroids, Vector* residuals, const int* assignments, int num_vecs, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vecs) return;

    int cluster = assignments[idx];
    for (int j = 0; j < dim; ++j) {
        residuals[idx].data[j] = vecs[idx].data[j] - centroids[cluster].data[j];
    }
}

// 残差量化函数
void residual_quantization(const std::vector<Vector>& vecs, const std::vector<Vector>& centroids, std::vector<Vector>& residuals, const std::vector<int>& assignments) {
    int num_vecs = vecs.size();
    int dim = DIM;

    Vector* d_vecs;
    Vector* d_centroids;
    Vector* d_residuals;
    int* d_assignments;

    // 分配CUDA内存
    CUDA_CHECK(cudaMalloc((void**)&d_vecs, num_vecs * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, centroids.size() * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_residuals, num_vecs * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, num_vecs * sizeof(int)));

    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_vecs, vecs.data(), num_vecs * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), centroids.size() * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_assignments, assignments.data(), num_vecs * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_vecs + blockSize - 1) / blockSize;

    residual_quantization_kernel<<<numBlocks, blockSize>>>(d_vecs, d_centroids, d_residuals, d_assignments, num_vecs, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制残差回主机
    CUDA_CHECK(cudaMemcpy(residuals.data(), d_residuals, num_vecs * sizeof(Vector), cudaMemcpyDeviceToHost));

    // 释放CUDA内存
    CUDA_CHECK(cudaFree(d_vecs));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_residuals));
    CUDA_CHECK(cudaFree(d_assignments));
}

// 近似最近邻搜索的CUDA核函数
__global__ void approximate_nearest_neighbors_kernel(const Vector* queries, const Vector* base, const Vector* residuals, const Vector* centroids, const int* assignments, int* results, int num_queries, int num_base, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;

    float min_dist = FLT_MAX;
    int best_index = -1;
    for (int i = 0; i < num_base; ++i) {
        float dist_to_centroid = 0;
        for (int d = 0; d < dim; ++d) {
            float diff = queries[idx].data[d] - centroids[assignments[i]].data[d];
            dist_to_centroid += diff * diff;
        }
        float dist = dist_to_centroid;
        for (int d = 0; d < dim; ++d) {
            float diff = queries[idx].data[d] - residuals[i].data[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_index = i;
        }
    }
    results[idx] = best_index;
}

// 近似最近邻搜索函数
std::vector<int> approximate_nearest_neighbors(const std::vector<Vector>& queries, const std::vector<Vector>& base, const std::vector<Vector>& residuals, const std::vector<Vector>& centroids, const std::vector<int>& assignments) {
    int num_queries = queries.size();
    int num_base = base.size();
    int dim = DIM;

    Vector* d_queries;
    Vector* d_base;
    Vector* d_residuals;
    Vector* d_centroids;
    int* d_assignments;
    int* d_results;

    // 分配CUDA内存
    CUDA_CHECK(cudaMalloc((void**)&d_queries, num_queries * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_base, num_base * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_residuals, num_base * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, centroids.size() * sizeof(Vector)));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, num_base * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, num_queries * sizeof(int)));

    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), num_queries * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base, base.data(), num_base * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residuals, residuals.data(), num_base * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), centroids.size() * sizeof(Vector), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_assignments, assignments.data(), num_base * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_queries + blockSize - 1) / blockSize;

    approximate_nearest_neighbors_kernel<<<numBlocks, blockSize>>>(d_queries, d_base, d_residuals, d_centroids, d_assignments, d_results, num_queries, num_base, K, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制结果回主机
    std::vector<int> results(num_queries);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, num_queries * sizeof(int), cudaMemcpyDeviceToHost));

    // 释放CUDA内存
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_residuals));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_results));

    return results;
}

int main() {
    // 读取基向量和查询向量
    std::vector<Vector> base_vecs;
    read_fvecs("sift-base.fvecs", base_vecs, BASE_VECS);
    std::vector<Vector> query_vecs;
    read_fvecs("sift-query.fvecs", query_vecs, QUERY_VECS);

    // 聚类
    std::vector<Vector> centroids(K);
    std::vector<int> assignments(base_vecs.size());
    kmeans(base_vecs, centroids, assignments, K);

    // 计算残差
    std::vector<Vector> residuals(base_vecs.size());
    residual_quantization(base_vecs, centroids, residuals, assignments);

    // 近似最近邻搜索
    std::vector<int> results = approximate_nearest_neighbors(query_vecs, base_vecs, residuals, centroids, assignments);

    // 输出结果
    for (int result : results) {
        std::cout << "最近邻索引: " << result << std::endl;
    }

    return 0;
}

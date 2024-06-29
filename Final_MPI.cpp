#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cfloat>
#include <random>

const int DIM = 128;        // 向量的维度
const int BASE_VECS = 200000;  // 读取的基向量数量
const int QUERY_VECS = 10000;  // 查询向量数量
const int K = 16;           // 聚类中心数量

struct Vector {
    float data[DIM];
};

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

float euclidean_distance(const Vector& a, const Vector& b) {
    float dist = 0;
    for (int i = 0; i < DIM; ++i) {
        float diff = a.data[i] - b.data[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

void kmeans(const std::vector<Vector>& vecs, std::vector<Vector>& centroids, std::vector<int>& assignments, int k, int max_iters = 100) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, vecs.size() - 1);

    for (int i = 0; i < k; ++i) {
        centroids.push_back(vecs[dis(gen)]);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        #pragma omp parallel for
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

        std::vector<Vector> new_centroids(k);
        std::vector<int> counts(k, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < vecs.size(); ++i) {
            int cluster = assignments[i];
            #pragma omp atomic
            for (int j = 0; j < DIM; ++j) {
                new_centroids[cluster].data[j] += vecs[i].data[j];
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
                new_centroids[j] = vecs[dis(gen)];
            }
        }
        centroids = new_centroids;
    }
}

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

std::vector<int> approximate_nearest_neighbors(const std::vector<Vector>& queries, const std::vector<Vector>& base, const std::vector<Vector>& residuals, const std::vector<Vector>& centroids, const std::vector<int>& assignments) {
    std::vector<int> results(queries.size());

    #pragma omp parallel for
    for (size_t q = 0; q < queries.size(); ++q) {
        float min_dist = FLT_MAX;
        int best_index = -1;
        for (size_t i = 0; i < base.size(); ++i) {
            float dist_to_centroid = euclidean_distance(queries[q], centroids[assignments[i]]);
            float dist = dist_to_centroid + euclidean_distance(queries[q], residuals[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_index = i;
            }
        }
        results[q] = best_index;
    }
    return results;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<Vector> base_vecs;
    std::vector<Vector> query_vecs;

    if (world_rank == 0) {
        read_fvecs("sift-base.fvecs", base_vecs, BASE_VECS);
        read_fvecs("sift-query.fvecs", query_vecs, QUERY_VECS);
    }

    int base_vecs_per_proc = BASE_VECS / world_size;
    int query_vecs_per_proc = QUERY_VECS / world_size;

    std::vector<Vector> local_base_vecs(base_vecs_per_proc);
    std::vector<Vector> local_query_vecs(query_vecs_per_proc);

    MPI_Scatter(base_vecs.data(), base_vecs_per_proc * DIM, MPI_FLOAT, local_base_vecs.data(), base_vecs_per_proc * DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(query_vecs.data(), query_vecs_per_proc * DIM, MPI_FLOAT, local_query_vecs.data(), query_vecs_per_proc * DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::vector<Vector> centroids;
    std::vector<int> assignments(base_vecs_per_proc);

    kmeans(local_base_vecs, centroids, assignments, K);

    std::vector<Vector> global_centroids(K);
    MPI_Allreduce(centroids.data(), global_centroids.data(), K * DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<Vector> residuals;
    residual_quantization(local_base_vecs, global_centroids, residuals, assignments);

    std::vector<int> local_results = approximate_nearest_neighbors(local_query_vecs, local_base_vecs, residuals, global_centroids, assignments);

    std::vector<int> global_results(QUERY_VECS);
    MPI_Gather(local_results.data(), query_vecs_per_proc, MPI_INT, global_results.data(), query_vecs_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int result : global_results) {
            std::cout << "最近邻索引: " << result << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

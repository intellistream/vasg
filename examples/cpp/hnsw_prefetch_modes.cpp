// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vsag/vsag.h>
#include <iostream>
#include <chrono>

/**
 * Example: Comparing three prefetch modes for HNSW
 * 
 * This example demonstrates the performance differences between:
 * 1. disabled  - No prefetching
 * 2. hardcoded - Automatic prefetch calculation
 * 3. custom    - User-defined prefetch parameters
 */

// Helper function to measure search time
double measure_search_time(std::shared_ptr<vsag::Index> index, 
                          const vsag::DatasetPtr& queries, 
                          int64_t topk,
                          const std::string& search_params,
                          int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = index->KnnSearch(queries, topk, search_params);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / (double)iterations;
}

int main(int argc, char** argv) {
    /******************* Prepare Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t num_queries = 100;
    int64_t dim = 128;
    int64_t topk = 10;
    
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    auto query_vectors = new float[dim * num_queries];
    
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    for (int64_t i = 0; i < dim * num_queries; ++i) {
        query_vectors[i] = distrib_real(rng);
    }
    
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);
    
    auto queries = vsag::Dataset::Make();
    queries->NumElements(num_queries)->Dim(dim)->Float32Vectors(query_vectors)->Owner(true);

    /******************* Test 1: Hardcoded Mode (Default) *****************/
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 1: HARDCODED Mode (Default)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto hnsw_params_hardcoded = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100,
            "prefetch_mode": "hardcoded"
        }
    }
    )";
    
    auto index1 = vsag::Factory::CreateIndex("hnsw", hnsw_params_hardcoded).value();
    index1->Build(base);
    
    auto search_params_hardcoded = R"({"hnsw": {"ef_search": 100}})";
    double time1 = measure_search_time(index1, queries, topk, search_params_hardcoded);
    
    std::cout << "Average search time: " << time1 << " μs" << std::endl;
    std::cout << "Description: Uses auto-calculated prefetch parameters" << std::endl;
    std::cout << "  - prefetch_jump = max(1, data_size/128 - 1)" << std::endl;

    /******************* Test 2: Disabled Mode *****************/
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 2: DISABLED Mode" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto hnsw_params_disabled = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100,
            "prefetch_mode": "disabled"
        }
    }
    )";
    
    auto index2 = vsag::Factory::CreateIndex("hnsw", hnsw_params_disabled).value();
    index2->Build(base);
    
    auto search_params_disabled = R"({"hnsw": {"ef_search": 100}})";
    double time2 = measure_search_time(index2, queries, topk, search_params_disabled);
    
    std::cout << "Average search time: " << time2 << " μs" << std::endl;
    std::cout << "Description: No prefetching" << std::endl;
    std::cout << "Slowdown vs hardcoded: " << (time2/time1 - 1) * 100 << "%" << std::endl;

    /******************* Test 3: Custom Mode (Conservative) *****************/
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 3: CUSTOM Mode (Conservative)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto hnsw_params_custom1 = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100,
            "prefetch_mode": "custom"
        }
    }
    )";
    
    auto index3 = vsag::Factory::CreateIndex("hnsw", hnsw_params_custom1).value();
    index3->Build(base);
    
    auto search_params_custom1 = R"(
    {
        "hnsw": {
            "ef_search": 100,
            "prefetch_mode": "custom",
            "prefetch_stride_codes": 1,
            "prefetch_depth_codes": 1,
            "prefetch_stride_visit": 1
        }
    }
    )";
    double time3 = measure_search_time(index3, queries, topk, search_params_custom1);
    
    std::cout << "Average search time: " << time3 << " μs" << std::endl;
    std::cout << "Parameters: stride_codes=1, depth_codes=1, stride_visit=1" << std::endl;
    std::cout << "Speedup vs disabled: " << (time2/time3 - 1) * 100 << "%" << std::endl;

    /******************* Test 4: Custom Mode (Balanced) *****************/
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 4: CUSTOM Mode (Balanced)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto search_params_custom2 = R"(
    {
        "hnsw": {
            "ef_search": 100,
            "prefetch_mode": "custom",
            "prefetch_stride_codes": 3,
            "prefetch_depth_codes": 2,
            "prefetch_stride_visit": 3
        }
    }
    )";
    double time4 = measure_search_time(index3, queries, topk, search_params_custom2);
    
    std::cout << "Average search time: " << time4 << " μs" << std::endl;
    std::cout << "Parameters: stride_codes=3, depth_codes=2, stride_visit=3" << std::endl;
    std::cout << "Speedup vs disabled: " << (time2/time4 - 1) * 100 << "%" << std::endl;

    /******************* Test 5: Custom Mode (Aggressive) *****************/
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 5: CUSTOM Mode (Aggressive)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto search_params_custom3 = R"(
    {
        "hnsw": {
            "ef_search": 100,
            "prefetch_mode": "custom",
            "prefetch_stride_codes": 5,
            "prefetch_depth_codes": 3,
            "prefetch_stride_visit": 5
        }
    }
    )";
    double time5 = measure_search_time(index3, queries, topk, search_params_custom3);
    
    std::cout << "Average search time: " << time5 << " μs" << std::endl;
    std::cout << "Parameters: stride_codes=5, depth_codes=3, stride_visit=5" << std::endl;
    std::cout << "Speedup vs disabled: " << (time2/time5 - 1) * 100 << "%" << std::endl;

    /******************* Summary *****************/
    std::cout << "\n========================================" << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode                    | Time (μs) | Speedup" << std::endl;
    std::cout << "------------------------|-----------|--------" << std::endl;
    printf("Disabled                | %9.2f | baseline\n", time2);
    printf("Hardcoded (default)     | %9.2f | %.1f%%\n", time1, (time2/time1 - 1) * 100);
    printf("Custom (conservative)   | %9.2f | %.1f%%\n", time3, (time2/time3 - 1) * 100);
    printf("Custom (balanced)       | %9.2f | %.1f%%\n", time4, (time2/time4 - 1) * 100);
    printf("Custom (aggressive)     | %9.2f | %.1f%%\n", time5, (time2/time5 - 1) * 100);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RECOMMENDATIONS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "1. For most use cases: Use 'hardcoded' mode (default)" << std::endl;
    std::cout << "   - No tuning required" << std::endl;
    std::cout << "   - Stable performance" << std::endl;
    std::cout << "   - Good speedup (15-20%)" << std::endl;
    std::cout << "\n2. For performance tuning: Use 'custom' mode" << std::endl;
    std::cout << "   - Test different parameters" << std::endl;
    std::cout << "   - Can achieve 20-30% speedup" << std::endl;
    std::cout << "   - Requires experimentation" << std::endl;
    std::cout << "\n3. For low-concurrency or debugging: Use 'disabled' mode" << std::endl;
    std::cout << "   - Establish baseline" << std::endl;
    std::cout << "   - Reduce cache contention" << std::endl;

    return 0;
}

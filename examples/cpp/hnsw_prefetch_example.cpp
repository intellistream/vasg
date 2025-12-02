
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

/**
 * Example: Using HNSW with custom prefetch optimization parameters
 * 
 * This example demonstrates how to configure prefetch optimization for HNSW index
 * to achieve better cache performance and reduce memory access latency.
 */

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create HNSW Index *****************/
    auto hnsw_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_parameters).value();

    /******************* Build HNSW Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HNSW contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* Prepare Query *****************/
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);
    int64_t topk = 10;

    /******************* Example 1: Default Search (no prefetch tuning) *****************/
    std::cout << "\n=== Example 1: Default Search ===" << std::endl;
    auto default_search_params = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    auto result1 = index->KnnSearch(query, topk, default_search_params);
    if (result1.has_value()) {
        std::cout << "Default search completed successfully" << std::endl;
    }

    /******************* Example 2: Conservative Prefetch (low cache usage) *****************/
    std::cout << "\n=== Example 2: Conservative Prefetch ===" << std::endl;
    auto conservative_search_params = R"(
    {
        "hnsw": {
            "ef_search": 100,
            "prefetch_stride_codes": 1,
            "prefetch_depth_codes": 1,
            "prefetch_stride_visit": 1
        }
    }
    )";
    auto result2 = index->KnnSearch(query, topk, conservative_search_params);
    if (result2.has_value()) {
        std::cout << "Conservative prefetch search completed" << std::endl;
        std::cout << "  - prefetch_stride_codes: 1 (minimal vector prefetching)" << std::endl;
        std::cout << "  - prefetch_depth_codes: 1 (64 bytes per prefetch)" << std::endl;
        std::cout << "  - prefetch_stride_visit: 1 (minimal visit prefetching)" << std::endl;
    }

    /******************* Example 3: Aggressive Prefetch (high cache usage) *****************/
    std::cout << "\n=== Example 3: Aggressive Prefetch ===" << std::endl;
    auto aggressive_search_params = R"(
    {
        "hnsw": {
            "ef_search": 100,
            "prefetch_stride_codes": 4,
            "prefetch_depth_codes": 3,
            "prefetch_stride_visit": 5
        }
    }
    )";
    auto result3 = index->KnnSearch(query, topk, aggressive_search_params);
    if (result3.has_value()) {
        std::cout << "Aggressive prefetch search completed" << std::endl;
        std::cout << "  - prefetch_stride_codes: 4 (prefetch 4 vectors ahead)" << std::endl;
        std::cout << "  - prefetch_depth_codes: 3 (192 bytes per prefetch)" << std::endl;
        std::cout << "  - prefetch_stride_visit: 5 (prefetch 5 nodes ahead)" << std::endl;
    }

    /******************* Example 4: Balanced Prefetch (recommended for SQ8) *****************/
    std::cout << "\n=== Example 4: Balanced Prefetch ===" << std::endl;
    auto balanced_search_params = R"(
    {
        "hnsw": {
            "ef_search": 100,
            "prefetch_stride_codes": 3,
            "prefetch_depth_codes": 2,
            "prefetch_stride_visit": 3
        }
    }
    )";
    auto result4 = index->KnnSearch(query, topk, balanced_search_params);
    if (result4.has_value()) {
        std::cout << "Balanced prefetch search completed" << std::endl;
        std::cout << "  - prefetch_stride_codes: 3 (moderate vector prefetching)" << std::endl;
        std::cout << "  - prefetch_depth_codes: 2 (128 bytes per prefetch)" << std::endl;
        std::cout << "  - prefetch_stride_visit: 3 (moderate visit prefetching)" << std::endl;
        std::cout << "  - Good for: quantized vectors (SQ8), medium dimensions" << std::endl;
    }

    /******************* Print Results from Last Search *****************/
    if (result4.has_value()) {
        auto result = result4.value();
        std::cout << "\nTop-" << topk << " results:" << std::endl;
        for (int64_t i = 0; i < std::min(topk, result->GetDim()); ++i) {
            std::cout << "  " << i+1 << ". ID: " << result->GetIds()[i] 
                      << ", Distance: " << result->GetDistances()[i] << std::endl;
        }
    }

    std::cout << "\n=== Prefetch Parameter Tuning Guidelines ===" << std::endl;
    std::cout << "prefetch_stride_codes: How many vectors to prefetch ahead" << std::endl;
    std::cout << "  - Low dimensional (<128): 3-5" << std::endl;
    std::cout << "  - Medium dimensional (128-512): 2-3" << std::endl;
    std::cout << "  - High dimensional (>512): 1-2" << std::endl;
    std::cout << "\nprefetch_depth_codes: Cache lines per vector (64 bytes each)" << std::endl;
    std::cout << "  - Formula: ceil(vector_bytes / 64)" << std::endl;
    std::cout << "  - FP32 128d: 512 bytes -> 8 lines (use 2-3 for partial)" << std::endl;
    std::cout << "  - SQ8 128d: 128 bytes -> 2 lines" << std::endl;
    std::cout << "\nprefetch_stride_visit: How many visited nodes to prefetch" << std::endl;
    std::cout << "  - Dense graph (high M): 3-5" << std::endl;
    std::cout << "  - Sparse graph (low M): 1-2" << std::endl;

    return 0;
}

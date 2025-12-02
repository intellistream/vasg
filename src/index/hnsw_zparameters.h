
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

#pragma once

#include <memory>
#include <string>

#include "algorithm/hnswlib/hnswlib.h"
#include "data_type.h"
#include "index_common_param.h"

namespace vsag {

// Prefetch optimization modes
enum class PrefetchMode {
    DISABLED = 0,    // No prefetching
    HARDCODED = 1,   // Use hardcoded prefetch_jump_code_size_ (auto-calculated)
    CUSTOM = 2       // Use user-defined prefetch parameters
};

struct HnswParameters {
public:
    static HnswParameters
    FromJson(const JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param);

public:
    // required vars
    std::shared_ptr<hnswlib::SpaceInterface> space;
    int64_t max_degree;
    int64_t ef_construction;
    bool use_conjugate_graph{false};
    bool use_static{false};
    bool normalize{false};
    bool use_reversed_edges{false};
    DataTypes type{DataTypes::DATA_TYPE_FLOAT};
    
    // prefetch optimization mode (set at build time)
    PrefetchMode prefetch_mode{PrefetchMode::HARDCODED};

protected:
    HnswParameters() = default;
};

struct FreshHnswParameters : public HnswParameters {
public:
    static HnswParameters
    FromJson(const JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param);

private:
    FreshHnswParameters() = default;
};

struct HnswSearchParameters {
public:
    static HnswSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    int64_t ef_search;
    float skip_ratio{0.9};
    bool use_conjugate_graph_search;
    
    // prefetch optimization mode (can override at search time)
    PrefetchMode prefetch_mode{PrefetchMode::HARDCODED};
    
    // custom prefetch parameters (only used when mode is CUSTOM)
    uint32_t prefetch_stride_codes{1};
    uint32_t prefetch_depth_codes{1};
    uint32_t prefetch_stride_visit{3};

private:
    HnswSearchParameters() = default;
};

}  // namespace vsag

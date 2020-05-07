// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum Mode { training, infer };

using SparseMeta =
    std::tuple<std::string, std::vector<std::string>, std::vector<int>, int>;

struct VALUE {
  void init() {
    for (int i = 0; i <= static_cast<int>(names.size()); i++) {
      values.reserve(dims[i]);
      std::fill(values_[i].data(), values_[i].size(), 0.0);
    }
  }

  void set() {}

  std::vector<std::vector<float>> get() {}

  std::vector<std::vector<float>> get(const std::vector<std::string> names) {}

  std::vector<std::vector<float>> Get() { return values; }

  std::vector<std::string> names_;
  std::vector<std::vector<float>> values_;
  std::vector<int> dims_;
  std::vector<int> initializer_;
};

class SparseVariable {
 public:
  SparseVariable();

  explicit SparseVariable(const SparseMeta& meta) {
    name_ = std::get<0>(meta);
    auto value_names = std::get<1>(meta);
    auto dims = std::get<2>(meta);
    auto mode = std::get<3>(meta);

    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      value_mata_[value_names[i]] = dims[i];
    }
    mode_ == mode == 0 ? Mode::training : Mode::infer;
  }

  void Get(const std::vector<int64_t>& ids,
           const std::vector<std::string>& value_names,
           std::vector<std::vector<std::vector<float>>>* values) {
    for (auto id : ids) {
      auto got = values_.find(id);
      if (got == values_.end()) {
        auto value = VALUE();
        value.init();
        values_[id] = value;
      }
      auto value = values_.at(id);
      values.push_back(value.get(value_names));
    }
  }

  void Get(const framework::Tensor& ids,
           const std::vector<std::string> value_names,
           std::vector<framework::Tensor>* values) {}

  int Size();

 private:
  std::string name_;
  Mode mode_;
  std::unordered_map<std::string, int> value_mata_;
  std::unordered_map<int64_t, VALUE> values_;
};

class LargeScaleKV {
 public:
  LargeScaleKV();
  ~LargeScaleKV();

  explicit LargeScaleKV(const std::vector<SparseMeta>& table_metas) {
    for (auto& sparse_meta : table_metas) {
      auto table_name = std::get<0>(sparse_meta);
      auto meta = std::make_shared<SparseVariable>(sparse_meta);
      sparse_variables[table_name] = meta;
    }
  }

  static LargeScaleKV* GetInstance() { return scale_kv_.get(); }

  static LargeScaleKV* InitInstance(
      const std::vector<SparseMeta>& table_metas) {
    std::call_once(init_flag_, &LargeScaleKV::Init, std::ref(table_metas));
    return scale_kv_.get();
  }

  static void Init(const std::vector<SparseMeta>& table_metas) {
    if (scale_kv_.get() == nullptr) {
      scale_kv_.reset(new LargeScaleKV(table_metas));
    }
  }

  SparseVariable* Get(std::string name) {
    auto variable = sparse_variables.at(name);
    return variable.get();
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<SparseVariable>>
      sparse_variables;
  static std::shared_ptr<LargeScaleKV> scale_kv_;
  static std::once_flag init_flag_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

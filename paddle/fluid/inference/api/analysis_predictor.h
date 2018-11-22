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
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/string/printf.h"
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif
namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using framework::NaiveExecutor;
using contrib::AnalysisConfig;

/* This predictor is based on the original native predictor with IR and Analysis
 * support. It will optimize IR and Parameters in the runtime.
 * TODO(Superjomn) Replace the Navive predictor?
 */
class AnalysisPredictor : public PaddlePredictor {
 public:
  explicit AnalysisPredictor(const AnalysisConfig &config) : config_(config) {}

  bool Init(const std::shared_ptr<framework::Scope> &parent_scope,
            const std::shared_ptr<framework::ProgramDesc> &program = nullptr);

  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;

  bool ZeroCopyRun() override;

  void CreateFeedFetchVar(framework::Scope *scope);
  void PrepareFeedFetch();

  void OptimizeInferenceProgram();

  Argument &analysis_argument() { return argument_; }

  std::unique_ptr<PaddlePredictor> Clone() override;

  framework::Scope *scope() { return scope_.get(); }
  framework::ProgramDesc &program() { return *inference_program_; }

 protected:
  void CollectVarShapes() {
    if (batch_var_shapes_.size() >= max_shape_collect_count_) return;
    std::map<std::string, std::vector<int>> var_shapes;
    for (auto var_name : inference_program_->Block(0).LocalVarNames()) {
      auto *var = sub_scope_->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(var);
      if (var->Type() == typeid(framework::LoDTensor) ||
          var->Type() == typeid(framework::Tensor)) {
        auto &tensor = var->Get<framework::LoDTensor>();
        auto shape = framework::vectorize(tensor.dims());
        var_shapes[var_name].assign(shape.begin(), shape.end());
      }
    }
    batch_var_shapes_.push_back(var_shapes);
    LOG(INFO) << "collect " << batch_var_shapes_.size()
              << " batch of var shapes for analysis";
  }

  void SerizlizeBatchVarShapes(const std::string &path) {
    LOG(INFO) << "serialize batch var shapes to " << path;
    std::ofstream file(path);
    if (!file.is_open()) {
      LOG(ERROR) << "failed to serialize the var shapes to " << path;
      return;
    }

    for (auto &batch : batch_var_shapes_) {
      for (auto &ele : batch) {
        file << ele.first << ":";
        for (int i = 0; i < ele.second.size() - 1; i++) {
          file << ele.second[i] << ",";
        }
        file << ele.second.back() << ";";
      }
      file << "\n";
    }
  }

  bool PrepareProgram(const std::shared_ptr<framework::ProgramDesc> &program);
  bool PrepareScope(const std::shared_ptr<framework::Scope> &parent_scope);
  bool CreateExecutor();
  bool PrepareExecutor();

  bool LoadProgramDesc();
  bool LoadParameters();

  bool SetFeed(const std::vector<PaddleTensor> &input_datas,
               framework::Scope *scope);
  bool GetFetch(std::vector<PaddleTensor> *output_data,
                framework::Scope *scope);
  template <typename T>
  void GetFetchOne(const framework::LoDTensor &fetchs,
                   PaddleTensor *output_data);
  ~AnalysisPredictor();

// Some more detailed tests, they are made the friends of the predictor, so that
// the all the details can be tested.
#if PADDLE_WITH_TESTING
  FRIEND_TEST(AnalysisPredictor, analysis_off);
  FRIEND_TEST(AnalysisPredictor, analysis_on);
  FRIEND_TEST(AnalysisPredictor, with_gpu);
#endif

 private:
  contrib::AnalysisConfig config_;
  Argument argument_;
  std::unique_ptr<NaiveExecutor> executor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_{nullptr};
  std::shared_ptr<framework::ProgramDesc> inference_program_;
  std::vector<framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  std::vector<framework::OpDesc *> fetchs_;
  // Memory buffer for feed inputs. The temporary LoDTensor will cause serious
  // concurrency problems, so cache them.
  std::vector<framework::LoDTensor> feed_tensors_;
  details::TensorArrayBatchCleaner tensor_array_batch_cleaner_;

  // Collect var shapes of batches.
  const size_t max_shape_collect_count_{1000};
  std::vector<std::map<std::string, std::vector<int>>> batch_var_shapes_;

 private:
  // Some status here that help to determine the status inside the predictor.
  bool status_program_optimized_{false};
  bool status_is_cloned_{false};
  bool status_use_gpu_{false};
  bool status_ir_optim_enabled_{false};
};

}  // namespace paddle

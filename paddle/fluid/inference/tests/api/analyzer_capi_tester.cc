/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  VLOG(3) << "data size " << record.data.size();
  VLOG(3) << "data shape size " << record.shape.size();
  return record;
}

const char *GetModelPath(std::string str) { return str.c_str(); }

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model/__model__",
                FLAGS_infer_model + "/model/__params__");
  cfg->DisableGpu();
  cfg->SwitchIrDebug();
  cfg->SwitchSpecifyInputNames(false);
  // TODO(TJ): fix fusion gru
  cfg->pass_builder()->DeletePass("fc_gru_fuse_pass");
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data, 0, "Only have single batch of data.");
  std::string line;
  std::ifstream file(FLAGS_infer_data);
  std::getline(file, line);
  auto record = ProcessALine(line);

  PaddleTensor input;
  input.shape = record.shape;
  input.dtype = PaddleDType::FLOAT32;
  size_t input_size = record.data.size() * sizeof(float);
  input.data.Resize(input_size);
  memcpy(input.data.data(), record.data.data(), input_size);
  std::vector<PaddleTensor> input_slots;
  input_slots.assign({input});
  (*inputs).emplace_back(input_slots);
}

void zero_copy_run() {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig config;
  PD_DisableGpu(&config);
  PD_SetCpuMathLibraryNumThreads(&config, 10);
  PD_SwitchUseFeedFetchOps(&config, false);
  PD_SwitchSpecifyInputNames(&config, true);
  PD_SwitchIrDebug(&config, true);
  PD_SetModel(&config, model_dir.c_str());  //, params_file1.c_str());
  bool use_feed_fetch = PD_UseFeedFetchOpsEnabled(&config);
  CHECK(!use_feed_fetch) << "NO";
  bool specify_input_names = PD_SpecifyInputName(&config);
  CHECK(specify_input_names) << "NO";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};
  int shape_size = 4;

  int in_size = 1;
  int *out_size;
  PD_ZeroCopyData *inputs = new PD_ZeroCopyData;
  PD_ZeroCopyData *outputs = new PD_ZeroCopyData;
  inputs->data = static_cast<void *>(input);
  inputs->dtype = PD_FLOAT32;
  inputs->name = new char[2];
  inputs->name[0] = 'x';
  inputs->name[1] = '\0';
  LOG(INFO) << inputs->name;
  inputs->shape = shape;
  inputs->shape_size = shape_size;

  PD_PredictorZeroCopyRun(&config, inputs, in_size, &outputs, &out_size);
}

TEST(PD_ZeroCopyRun, zero_copy_run) { zero_copy_run(); }

TEST(PD_AnalysisConfig, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  int num_thread = PD_CpuMathLibraryNumThreads(config);
  CHECK(10 == num_thread) << "NO";
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, model_dir.c_str());
  PD_SetOptimCacheDir(config, (FLAGS_infer_model + "/OptimCacheDir").c_str());
  const char *model_dir_ = PD_ModelDir(config);
  LOG(INFO) << model_dir_;
  PD_EnableUseGpu(config, 100, 0);
  bool use_gpu = PD_UseGpu(config);
  CHECK(use_gpu) << "NO";
  int device = PD_GpuDeviceId(config);
  CHECK(0 == device) << "NO";
  int init_size = PD_MemoryPoolInitSizeMb(config);
  CHECK(100 == init_size) << "NO";
  float frac = PD_FractionOfGpuMemoryForPool(config);
  LOG(INFO) << frac;
  PD_EnableCUDNN(config);
  bool cudnn = PD_CudnnEnabled(config);
  CHECK(cudnn) << "NO";
  PD_SwitchIrOptim(config, true);
  bool ir_optim = PD_IrOptim(config);
  CHECK(ir_optim) << "NO";
  PD_EnableTensorRtEngine(config);
  bool trt_enable = PD_TensorrtEngineEnabled(config);
  CHECK(trt_enable) << "NO";
  /*PD_EnableAnakinEngine(config, );
  bool anakin_enable = PD_AnakinEngineEnabled(config);*/
  PD_EnableNgraph(config);
  bool ngraph_enable = PD_NgraphEnabled(config);
  CHECK(ngraph_enable) << "NO";
  PD_EnableMemoryOptim(config);
  bool memory_optim_enable = PD_MemoryOptimEnabled(config);
  CHECK(memory_optim_enable) << "NO";
  PD_EnableProfile(config);
  bool profiler_enable = PD_ProfileEnabled(config);
  CHECK(profiler_enable) << "NO";
  PD_SetInValid(config);
  bool is_valid = PD_IsValid(config);
  CHECK(is_valid) << "NO";
  PD_DeleteAnalysisConfig(config);
}

#ifdef PADDLE_WITH_MKLDNN
TEST(PD_AnalysisConfig, profile_mkldnn) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_EnableMKLDNN(config);
  bool mkldnn_enable = PD_MkldnnEnabled(config);
  CHECK(mkldnn_enable) << "NO";
  PD_EnableMkldnnQuantizer(config);
  bool quantizer_enable = PD_MkldnnQuantizerEnabled(config);
  CHECK(quantizer_enable) << "NO";
  PD_SetModel(config, model_dir.c_str());
  PD_DeleteAnalysisConfig(config);
}
#endif

// Check the fuse status
// TEST(Analyzer_vis, fuse_statis) {
//   AnalysisConfig cfg;
//   SetConfig(&cfg);
//   int num_ops;
//   auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
//   GetFuseStatis(predictor.get(), &num_ops);
// }

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// TEST(Analyzer_vis, compare) { compare(); }
// #ifdef PADDLE_WITH_MKLDNN
// TEST(Analyzer_vis, compare_mkldnn) { compare(true /* use_mkldnn */); }
// #endif

// // Compare Deterministic result
// TEST(Analyzer_vis, compare_determine) {
//   AnalysisConfig cfg;
//   SetConfig(&cfg);

//   std::vector<std::vector<PaddleTensor>> input_slots_all;
//   SetInput(&input_slots_all);
//   CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config
//   *>(&cfg),
//                        input_slots_all);
// }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

/*namespace paddle {
namespace inference {

const char* GetModelPath(std::string a) { return a.c_str(); }


TEST(Analysis_capi, compare) {
  std::string a = FLAGS_infer_model;
  const char* model_dir =
      GetModelPath(FLAGS_infer_model + "/mobilenet/__model__");
  const char* params_file =
      GetModelPath(FLAGS_infer_model + "/mobilenet/__params__");
  LOG(INFO) << model_dir;
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  PD_SetModel(config, model_dir, params_file);
  LOG(INFO) << PD_ModelDir(config);
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  LOG(INFO) << "before here! ";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};

  AnalysisConfig c;
  c.SetModel(model_dir, params_file);
  LOG(INFO) << c.model_dir();
  c.DisableGpu();
  c.SwitchUseFeedFetchOps(false);
  int shape_size = 4;
  auto predictor = CreatePaddlePredictor(c);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  std::vector<int> tensor_shape;
  tensor_shape.assign(shape, shape + shape_size);
  input_t->Reshape(tensor_shape);
  input_t->copy_from_cpu(input);
  CHECK(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle*/

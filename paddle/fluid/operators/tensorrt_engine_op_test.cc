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
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

USE_CPU_ONLY_OP(tensorrt_engine);

namespace paddle {
namespace operators {

namespace {
void CreateCPUTensor(framework::Scope* scope, const std::string& name,
                     const std::vector<int64_t>& shape) {
  auto* var = scope->Var(name);
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  auto dims = framework::make_ddim(shape);
  tensor->Resize(dims);
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  inference::tensorrt::RandomizeTensor(tensor, place, ctx);
}

void AddTensorToBlockDesc(framework::proto::BlockDesc* block,
                          const std::string& name,
                          const std::vector<int64_t>& shape) {
  using framework::proto::VarType;
  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::LOD_TENSOR);
  desc.SetDataType(VarType::FP32);
  desc.SetShape(shape);
  *var = *desc.Proto();
}

template <typename T>
void SetAttr(framework::proto::OpDesc* op, const std::string& name,
             const T& data);

template <>
void SetAttr<std::string>(framework::proto::OpDesc* op, const std::string& name,
                          const std::string& data) {
  auto* attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::STRING);
  attr->set_s(data);
}
template <>
void SetAttr<int>(framework::proto::OpDesc* op, const std::string& name,
                  const int& data) {
  auto* attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::INT);
  attr->set_i(data);
}
template <>
void SetAttr<int64_t>(framework::proto::OpDesc* op, const std::string& name,
                      const int64_t& data) {
  auto* attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::LONG);
  attr->set_l(data);
}
template <>
void SetAttr<std::vector<std::string>>(framework::proto::OpDesc* op,
                                       const std::string& name,
                                       const std::vector<std::string>& data) {
  auto* attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::STRINGS);
  for (const auto& s : data) {
    attr->add_strings(s.c_str());
  }
}

}  // namespace

// TEST(TensorRTEngineOp, manual) {
//   framework::ProgramDesc program;
//   auto* block_ = program.Proto()->add_blocks();
//   block_->set_idx(0);
//   block_->set_parent_idx(-1);

//   LOG(INFO) << "create block desc";
//   framework::BlockDesc block_desc(&program, block_);
//   LOG(INFO) << "create mul op";
//   auto* mul = block_desc.AppendOp();
//   mul->SetType("mul");
//   mul->SetInput("X", std::vector<std::string>({"x"}));     // 2 x 4
//   mul->SetInput("Y", std::vector<std::string>({"y"}));     // 4 x 6
//   mul->SetOutput("Out", std::vector<std::string>({"z"}));  // 2 x 6

//   LOG(INFO) << "create fc op";
//   auto* fc = block_desc.AppendOp();
//   fc->SetType("mul");
//   fc->SetInput("X", std::vector<std::string>({"z"}));
//   fc->SetInput("Y", std::vector<std::string>({"y0"}));     // 6 x 8
//   fc->SetOutput("Out", std::vector<std::string>({"z0"}));  // 2 x 8

//   // Set inputs' variable shape in BlockDesc
//   AddTensorToBlockDesc(block_, "x", std::vector<int64_t>({2, 4}));
//   AddTensorToBlockDesc(block_, "y", std::vector<int64_t>({4, 6}));
//   AddTensorToBlockDesc(block_, "y0", std::vector<int64_t>({6, 8}));
//   AddTensorToBlockDesc(block_, "z", std::vector<int64_t>({2, 6}));

//   // It is wired, need to copy manually.
//   *block_->add_ops() = *mul->Proto();
//   *block_->add_ops() = *fc->Proto();

//   ASSERT_EQ(block_->ops_size(), 2);

//   LOG(INFO) << "create tensorrt desc";
//   framework::OpDesc engine_op_desc(nullptr);
//   engine_op_desc.SetType("tensorrt_engine");
//   engine_op_desc.SetInput("Xs", std::vector<std::string>({"x", "y", "y0"}));
//   engine_op_desc.SetOutput("Ys", std::vector<std::string>({"z0"}));
//   SetAttr<std::string>(engine_op_desc.Proto(), "subgraph",
//                        block_->SerializeAsString());
//   SetAttr<int>(engine_op_desc.Proto(), "max_batch", 100);
//   SetAttr<int>(engine_op_desc.Proto(), "max_workspace", 1 << 10);

//   LOG(INFO) << "create engine op";
//   auto engine_op = framework::OpRegistry::CreateOp(*engine_op_desc.Proto());

//   framework::Scope scope;
//   platform::CPUPlace place;
//   platform::CPUDeviceContext ctx(place);
//   // Prepare variables.
//   CreateCPUTensor(&scope, "x", std::vector<int64_t>({2, 4}));
//   CreateCPUTensor(&scope, "y", std::vector<int64_t>({4, 6}));
//   CreateCPUTensor(&scope, "z", std::vector<int64_t>({2, 6}));

//   CreateCPUTensor(&scope, "y0", std::vector<int64_t>({6, 8}));
//   CreateCPUTensor(&scope, "z0", std::vector<int64_t>({2, 8}));

//   // Execute them.
//   LOG(INFO) << "engine_op run";
//   engine_op->Run(scope, place);
// }

void Execute(int batch_size, int input_dim, int output_dim, int nlayers = 1) {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);

  auto* block_ = program.Proto()->add_blocks();
  block_->set_idx(0);
  block_->set_parent_idx(-1);

  LOG(INFO) << "create block desc";
  framework::BlockDesc block_desc(&program, block_);

  LOG(INFO) << "create fc op";
  auto* fc = block_desc.AppendOp();
  fc->SetType("mul");
  fc->SetInput("X", std::vector<std::string>({"x"}));
  fc->SetInput("Y", std::vector<std::string>({"y"}));
  fc->SetOutput("Out", std::vector<std::string>({"z"}));

  // Set inputs' variable shape in BlockDesc
  AddTensorToBlockDesc(block_, "x",
                       std::vector<int64_t>({batch_size, input_dim}));
  AddTensorToBlockDesc(block_, "y",
                       std::vector<int64_t>({input_dim, output_dim}));
  AddTensorToBlockDesc(block_, "z",
                       std::vector<int64_t>({batch_size, output_dim}));

  // Prepare variables.
  CreateCPUTensor(&scope, "x", std::vector<int64_t>({batch_size, input_dim}));
  CreateCPUTensor(&scope, "y", std::vector<int64_t>({input_dim, output_dim}));
  CreateCPUTensor(&scope, "z", std::vector<int64_t>({batch_size, output_dim}));

  // It is wired, need to copy manually.
  *block_->add_ops() = *fc->Proto();

  ASSERT_EQ(block_->ops_size(), 1);

  LOG(INFO) << "create tensorrt desc";
  framework::OpDesc engine_op_desc(nullptr);
  engine_op_desc.SetType("tensorrt_engine");
  engine_op_desc.SetInput("Xs", std::vector<std::string>({"x", "y"}));
  engine_op_desc.SetOutput("Ys", std::vector<std::string>({"z"}));

  SetAttr<std::string>(engine_op_desc.Proto(), "subgraph",
                       block_->SerializeAsString());
  SetAttr<int>(engine_op_desc.Proto(), "max_batch", batch_size);
  SetAttr<int>(engine_op_desc.Proto(), "max_workspace", 1 << 10);
  SetAttr<std::vector<std::string>>(engine_op_desc.Proto(), "parameters",
                                    std::vector<std::string>({"y"}));
  // SetAttr<std::vector<std::string>>(engine_op_desc.Proto(), "parameters",
  // {});

  LOG(INFO) << "create engine op";
  auto engine_op = framework::OpRegistry::CreateOp(*engine_op_desc.Proto());

  // Execute them.
  LOG(INFO) << "engine_op run";
  engine_op->Run(scope, place);
}

TEST(fc, manulal) { Execute(100, 60, 50); }

}  // namespace operators
}  // namespace paddle

USE_TRT_CONVERTER(mul)
USE_TRT_CONVERTER(fc)

/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/selected_rows_functor.h"
#include "gtest/gtest.h"
#include "paddle/operators/math/math_function.h"

TEST(selected_rows_functor, cpu_add) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  using namespace paddle::operators::math;

  CPUPlace cpu_place;
  CPUDeviceContext ctx(cpu_place);
  SetConstant<CPUPlace, float> functor;
  int64_t height = 10;
  int64_t row_numel = 10;

  std::vector<int64_t> rows1{0, 4, 7};
  std::unique_ptr<SelectedRows> selected_rows1{new SelectedRows(rows1, height)};
  auto* in1_value = selected_rows1->mutable_value();
  in1_value->mutable_data<float>(
      make_ddim({static_cast<int64_t>(rows1.size()), row_numel}), cpu_place);
  functor(ctx, in1_value, 1.0);

  std::vector<int64_t> rows2{0, 5, 7, 9};
  std::unique_ptr<SelectedRows> selected_rows2{new SelectedRows(rows2, height)};
  auto* in2_value = selected_rows2->mutable_value();
  in2_value->mutable_data<float>(
      make_ddim({static_cast<int64_t>(rows2.size()), row_numel}), cpu_place);
  functor(ctx, in2_value, 2.0);

  std::unique_ptr<SelectedRows> output{new SelectedRows()};
  auto* out_value = output->mutable_value();

  // simplely concat two SelectedRows
  out_value->mutable_data<float>(make_ddim({7, 10}), cpu_place);

  SelectedRowsAdd<CPUPlace, float> add_functor;
  add_functor(ctx, *selected_rows1, *selected_rows2, output.get());

  auto out_height = output->height();
  EXPECT_EQ(out_height, height);

  auto& out_rows = output->rows();

  // input1 rows
  EXPECT_EQ(out_rows[0], 0);
  EXPECT_EQ(out_rows[1], 4);
  EXPECT_EQ(out_rows[2], 7);
  // input2 rows
  EXPECT_EQ(out_rows[3], 0);
  EXPECT_EQ(out_rows[4], 5);
  EXPECT_EQ(out_rows[5], 7);
  EXPECT_EQ(out_rows[6], 9);

  auto* out_data = output->value().data<float>();
  // input1 value
  EXPECT_EQ(out_data[0 * row_numel + 0], 1.0);
  EXPECT_EQ(out_data[0 * row_numel + 8], 1.0);
  EXPECT_EQ(out_data[1 * row_numel + 1], 1.0);
  EXPECT_EQ(out_data[2 * row_numel + 6], 1.0);
  // input2 value
  EXPECT_EQ(out_data[3 * row_numel + 3], 2.0);
  EXPECT_EQ(out_data[3 * row_numel + 8], 2.0);
  EXPECT_EQ(out_data[4 * row_numel + 4], 2.0);
  EXPECT_EQ(out_data[5 * row_numel + 7], 2.0);
  EXPECT_EQ(out_data[6 * row_numel + 9], 2.0);

  std::unique_ptr<Tensor> tensor1{new Tensor()};
  tensor1->mutable_data<float>(make_ddim({height, row_numel}), cpu_place);
  functor(ctx, tensor1.get(), 3.0);

  std::unique_ptr<Tensor> tensor2{new Tensor()};
  tensor2->mutable_data<float>(make_ddim({height, row_numel}), cpu_place);

  SelectedRowsAddTensor<CPUPlace, float> add_tensor_functor;
  add_tensor_functor(ctx, *output, *tensor1, tensor2.get());

  auto* tensor2_data = tensor2->data<float>();
  // row0: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor2_data[0 * row_numel + 0], 6.0);
  // row1: 3.0
  EXPECT_EQ(tensor2_data[1 * row_numel + 1], 3.0);
  // row4 : 1.0 + 3.0
  EXPECT_EQ(tensor2_data[4 * row_numel + 6], 4.0);
  // row5: 2.0 + 3.0
  EXPECT_EQ(tensor2_data[5 * row_numel + 7], 5.0);
  // row6: 3.0
  EXPECT_EQ(tensor2_data[6 * row_numel + 1], 3.0);
  // row7: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor2_data[7 * row_numel + 3], 6.0);
  // row9: 2.0 + 3.0
  EXPECT_EQ(tensor2_data[9 * row_numel + 6], 5.0);
}

TEST(selected_rows_functor, cpu_add_to) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  using namespace paddle::operators::math;

  CPUPlace cpu_place;
  CPUDeviceContext ctx(cpu_place);
  SetConstant<CPUPlace, float> functor;
  int64_t height = 10;
  int64_t row_numel = 10;

  std::vector<int64_t> rows1{0, 4, 7};
  std::unique_ptr<SelectedRows> selected_rows1{new SelectedRows(rows1, height)};
  auto* in1_value = selected_rows1->mutable_value();
  in1_value->mutable_data<float>(
      make_ddim({static_cast<int64_t>(rows1.size()), row_numel}), cpu_place);
  functor(ctx, in1_value, 1.0);

  std::vector<int64_t> rows2{0, 5, 7, 9};
  std::unique_ptr<SelectedRows> selected_rows2{new SelectedRows(rows2, height)};
  auto* in2_value = selected_rows2->mutable_value();
  in2_value->mutable_data<float>(
      make_ddim({static_cast<int64_t>(rows2.size()), row_numel}), cpu_place);
  functor(ctx, in2_value, 2.0);

  std::unique_ptr<SelectedRows> output{new SelectedRows()};
  output->set_height(height);
  auto* out_value = output->mutable_value();

  // simplely concat two SelectedRows
  out_value->mutable_data<float>(make_ddim({7, 10}), cpu_place);

  SelectedRowsAddTo<CPUPlace, float> add_to_functor;
  add_to_functor(ctx, *selected_rows1, 0, output.get());
  add_to_functor(ctx, *selected_rows2, in1_value->numel(), output.get());

  auto out_height = output->height();
  EXPECT_EQ(out_height, height);

  auto& out_rows = output->rows();

  // input1 rows
  EXPECT_EQ(out_rows[0], 0);
  EXPECT_EQ(out_rows[1], 4);
  EXPECT_EQ(out_rows[2], 7);
  // input2 rows
  EXPECT_EQ(out_rows[3], 0);
  EXPECT_EQ(out_rows[4], 5);
  EXPECT_EQ(out_rows[5], 7);
  EXPECT_EQ(out_rows[6], 9);

  auto* out_data = output->value().data<float>();
  // input1 value
  EXPECT_EQ(out_data[0 * row_numel + 0], 1.0);
  EXPECT_EQ(out_data[0 * row_numel + 8], 1.0);
  EXPECT_EQ(out_data[1 * row_numel + 1], 1.0);
  EXPECT_EQ(out_data[2 * row_numel + 6], 1.0);
  // input2 value
  EXPECT_EQ(out_data[3 * row_numel + 3], 2.0);
  EXPECT_EQ(out_data[3 * row_numel + 8], 2.0);
  EXPECT_EQ(out_data[4 * row_numel + 4], 2.0);
  EXPECT_EQ(out_data[5 * row_numel + 7], 2.0);
  EXPECT_EQ(out_data[6 * row_numel + 9], 2.0);

  std::unique_ptr<Tensor> tensor1{new Tensor()};
  tensor1->mutable_data<float>(make_ddim({height, row_numel}), cpu_place);
  functor(ctx, tensor1.get(), 3.0);

  SelectedRowsAddToTensor<CPUPlace, float> add_to_tensor_functor;
  add_to_tensor_functor(ctx, *output, tensor1.get());

  auto* tensor1_data = tensor1->data<float>();
  // row0: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor1_data[0 * row_numel + 0], 6.0);
  // row1: 3.0
  EXPECT_EQ(tensor1_data[1 * row_numel + 1], 3.0);
  // row4 : 1.0 + 3.0
  EXPECT_EQ(tensor1_data[4 * row_numel + 6], 4.0);
  // row5: 2.0 + 3.0
  EXPECT_EQ(tensor1_data[5 * row_numel + 7], 5.0);
  // row6: 3.0
  EXPECT_EQ(tensor1_data[6 * row_numel + 1], 3.0);
  // row7: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor1_data[7 * row_numel + 3], 6.0);
  // row9: 2.0 + 3.0
  EXPECT_EQ(tensor1_data[9 * row_numel + 6], 5.0);
}

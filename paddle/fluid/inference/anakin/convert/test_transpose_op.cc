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
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(transpose_op, test) {
  auto* converter = Registry<AnakinOpConverter>::Global().Lookup("transpose");
  ASSERT_TRUE(converter != nullptr);
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, scope);
  validator.DeclInputVar("transpose-X", {2, 3, 4, 5});
  validator.DeclOutputVar("transpose-Out", {4, 2, 5, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("transpose");
  desc.SetInput("X", {"transpose-X"});
  desc.SetOutput("Output", {"transpose-Out"});
  desc.SetAttr<std::vector<int>>("axis", {2, 0, 3, 1});

  validator.SetOp(*desc.Proto());

  validator.Execute(3);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(transpose);
USE_ANAKIN_CONVERTER(transpose);

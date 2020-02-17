/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/edit_distance_op.h"

namespace paddle {
namespace operators {

class EditDistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Hyps"), "Input(Hyps) sholdn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Refs"), "Input(Refs) sholdn't be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) sholdn't be null.");
    PADDLE_ENFORCE(ctx->HasOutput("SequenceNum"),
                   "Output(SequenceNum) sholdn't be null.");
    auto hyp_dims = ctx->GetInputDim("Hyps");
    auto ref_dims = ctx->GetInputDim("Refs");

    if (ctx->HasInput("HypsLength") && ctx->HasInput("RefsLength")) {
      auto hyp_length_dims = ctx->GetInputDim("HypsLength");
      auto ref_length_dims = ctx->GetInputDim("RefsLength");

      PADDLE_ENFORCE(hyp_dims.size() == 2 && ref_dims.size() == 2 &&
                         hyp_dims[0] == ref_dims[0],
                     "Input(Hyps) and Input(Refs) must be 2-D Tensors with "
                     "identical first dimension");
      PADDLE_ENFORCE(hyp_length_dims[0] == ref_length_dims[0] &&
                         hyp_length_dims[0] == hyp_dims[0],
                     "Input(HypsLength), Input(RefsLength) and Input(Hyps) "
                     "shold have identical first dimension");
    } else {
      PADDLE_ENFORCE(
          hyp_dims.size() == 2 && hyp_dims[1] == 1,
          "Input(Hyps) must be a 2-D LoDTensor with the 2nd dimension "
          "equal to 1.");
      PADDLE_ENFORCE(
          ref_dims.size() == 2 && ref_dims[1] == 1,
          "Input(Refs) must be a 2-D LoDTensor with the 2nd dimension "
          "equal to 1.");
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("Refs"));
    ctx->SetOutputDim("SequenceNum", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class EditDistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Hyps",
             "2-D Tensor<int64_t>, or 2-D LoDTensor<int64_t> with last "
             "dimension being 1. "
             "The indices for hypothesis strings.");
    AddInput("Refs",
             "2-D Tensor<int64_t>, or 2-D LoDTensor<int64_t> with last "
             "dimension being 1. "
             "The indices for reference strings.");
    AddInput("HypsLength",
             "1-D Tensor<int64_t>. "
             "Sequence length for hyps when hyps is a tensor")
        .AsDispensable();
    AddInput("RefsLength",
             "1-D Tensor<int64_t>. "
             "Sequence length for refs when refs is a tensor")
        .AsDispensable();
    AddOutput("SequenceNum", "The sequence count of current batch");
    AddAttr<bool>("normalized",
                  "(bool, default false) Indicated whether to normalize "
                  "the edit distance by the length of reference string.")
        .SetDefault(false);
    AddOutput("Out",
              "(2-D Tensor with shape [`batch_size` x 1]) "
              "The output edit distances of EditDistance operator.");
    AddComment(R"DOC(

EditDistance operator computes the edit distances between a batch of hypothesis
strings and their references.

Edit distance, also called Levenshtein distance, measures how dissimilar two strings
are by counting the minimum number of operations to transform one string into another.
The operations include insertion, deletion, and substitution. 

For example, given hypothesis string A = "kitten" and reference B = "sitting",
A will be transformed into B at least after two substitutions and one
insertion:

   "kitten" -> "sitten" -> "sittin" -> "sitting"

So the edit distance between A and B is 3.

Input(Hyps) is a 2-D Tensor or a 2-D LoDTensor consisting of all the hypothesis strings.
And the `batch_size` reference strings are arranged in order in the same way in the
Input(Refs).

Output(Out) contains the `batch_size` results and each stands for the edit distance
for a pair of strings respectively. If Attr(normalized) is true, the edit distance
will be divided by the length of reference string.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    edit_distance, ops::EditDistanceOp, ops::EditDistanceOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    edit_distance, ops::EditDistanceKernel<paddle::platform::CPUPlace, float>);

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/channel.h"

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kOutputs[] = "Out";

class ChannelRecvOp : public framework::OperatorBase {
public:
    ChannelRecvOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
            : framework::OperatorBase(type, inputs, outputs, attrs) {}

private:
    void RunImpl(const framework::Scope &scope,
                 const platform::Place &dev_place) const override {
    }
};

class ChannelRecvOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    ChannelRecvOpMaker(OpProto *proto, OpAttrChecker *op_checker)
            : OpProtoAndCheckerMaker(proto, op_checker) {
      AddInput(kX,
               "A set of variables, which are required by operators inside the "
                       "block of Go Op.")
              .AsDuplicable();
      AddOutput(kOutputs,
                "A set of variables, which will be assigned with values "
                        "generated by the operators inside the block of Go Op.")
              .AsDuplicable();
      AddComment(R"DOC(
)DOC");
    }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(channel_recv, paddle::operators::ChannelRecvOp,
        paddle::framework::EmptyGradOpMaker,
        paddle::operators::ChannelRecvOpMaker);

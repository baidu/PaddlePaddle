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

#include <ostream>

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

#include "paddle/operators/detail/grpc_client.h"
#include "paddle/operators/detail/send_recv_impl.h"
#include "paddle/operators/detail/simple_block_queue.h"

namespace paddle {
namespace operators {

// TODO(gongwb): add more attrs to support more send pattern.
class SendOp : public framework::OperatorBase {
 public:
  SendOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    // init client when the operator is created at runtime.
    std::vector<std::string> endpoints =
        Attr<std::vector<std::string>>("endpoints");

    client_.AddEndPoint(endpoints);
  }

  void LogErrors(const std::vector<detail::SendStatus> &status) const {
    for (auto &s : status) {
      if (s.error != "") {
        LOG(ERROR) << "sync update variable error:" << s.String();
      }
    }
  }

  void Run(const framework::Scope &scope,
           const platform::Place &dev_place) const override {
    auto ins = Inputs("X");
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");

    std::vector<detail::VarHandle> in_vars;
    std::vector<detail::VarHandle> out_vars;
    for (int i = 0; i < int(ins.size()); i++) {
      detail::VarHandle in_var;
      in_var.name = ins[i];
      in_var.endpoint = epmap[i];
      in_vars.push_back(in_var);

      detail::VarHandle out_var;
      in_var.name = ins[i];
      in_var.endpoint = epmap[i];
      out_vars.push_back(out_var);
    }

    std::vector<detail::SendStatus> in_status;
    std::vector<detail::SendStatus> out_status;
    bool ok =
        client_.SyncUpdate(&scope, in_vars, in_status, out_vars, out_status);
    if (!ok) {
      LogErrors(in_status);
      LogErrors(out_status);
    }
  }

 protected:
  mutable detail::AsyncGRPCClient client_;
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be send").AsDuplicable();
    AddOutput("Out", "(Tensor) Output tensor to get from server")
        .AsDuplicable();
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send, ops::SendOp, ops::SendOpMaker);

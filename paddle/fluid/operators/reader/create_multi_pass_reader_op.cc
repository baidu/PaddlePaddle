//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class MultiPassReader : public framework::DecoratedReader {
 public:
  MultiPassReader(ReaderBase* reader, int pass_num)
      : DecoratedReader(reader), pass_num_(pass_num), cur_pass_(0) {}

  void ReInit() override {
    cur_pass_ = 0;
    reader_->ReInit();
  }

  LoDTensorListPtr ReadNext() override {
    LoDTensorListPtr res = nullptr;
    while (true) {
      res = ReadNext();
      if (res != nullptr) {
        break;
      }
      ++cur_pass_;
      if (cur_pass_ < pass_num_) {
        reader_->ReInit();
      }
    }
    return std::move(res);
  }

 private:
  int pass_num_;
  mutable int cur_pass_;
};

class CreateMultiPassReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = detail::Ref(scope.FindVar(Output("Out")))
                    .GetMutable<framework::ReaderHolder>();
    if (out->Get() != nullptr) {
      return;
    }
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    int pass_num = Attr<int>("pass_num");
    out->Reset(std::unique_ptr<framework::ReaderBase>(
        new MultiPassReader(underlying_reader.Get(), pass_num)));
  }
};

class CreateMultiPassReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateMultiPassReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddAttr<int>("pass_num", "The number of pass to run.").GreaterThan(0);
    AddComment(R"DOC(
      CreateMultiPassReader Operator

      This operator creates a multi-pass reader. A multi-pass reader 
      is used to yield data for several pass training continuously. 
      It takes the number of passes to run as one of its attributes
      ('pass_num'), and maintains a pass counter to record how many 
      passes it has completed. When the underlying reader reaches the 
      EOF, the multi-pass reader checks whether it has completed training 
      of the given number of pass. If not, the underlying reader will 
      be re-initialized and starts a new pass automatically.
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_multi_pass_reader,
                                   ops::CreateMultiPassReaderOp,
                                   ops::CreateMultiPassReaderOpMaker);

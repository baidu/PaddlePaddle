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

#include <stdint.h>
#include <sys/stat.h>
#include <fstream>
#include <numeric>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

// TODO(yuyang18): If the functions below are needed by other files, move them
// to paddle::filesystem namespace.
constexpr char kSEP = '/';
static bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static void MkDir(const char *path) {
  if (mkdir(path, 0755)) {
    PADDLE_ENFORCE_EQ(errno, EEXIST, "%s mkdir failed!", path);
  }
}

static void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}

class SaveOp : public framework::OperatorBase {
 public:
  SaveOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto filename = Attr<std::string>("file_path");
    auto overwrite = Attr<bool>("overwrite");

    if (FileExists(filename) && !overwrite) {
      PADDLE_THROW("%s is existed, cannot save to it when overwrite=false",
                   filename, overwrite);
    }

    MkDirRecursively(DirName(filename).c_str());

    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    std::ofstream fout(filename);
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   filename);

    auto iname = Input("X");
    auto *var = scope.FindVar(iname);
    PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s for save_op",
                   iname);

    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "SaveOp only support LoDTensor, %s has wrong type", iname);

    auto &tensor = var->Get<framework::LoDTensor>();

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    auto in_dtype =
        static_cast<framework::proto::VarType::Type>(Attr<int>("in_dtype"));
    auto out_dtype =
        static_cast<framework::proto::VarType::Type>(Attr<int>("out_dtype"));

    PADDLE_ENFORCE_EQ(
        static_cast<int>(framework::ToDataType(tensor.type())),
        static_cast<int>(in_dtype),
        "the tensor dtype does not match the attr of the save op");

    std::cout << std::endl
              << "filename is " << filename << ", var name is " << iname
              << std::endl
              << ", in_dtype is " << static_cast<int>(in_dtype)
              << ", out_dtype is " << static_cast<int>(out_dtype) << std::endl;

    std::cout << "before the conversion or not, the dtype is "
              << static_cast<int>(framework::ToDataType(tensor.type()))
              << std::endl;

    if (in_dtype != out_dtype) {
      std::cout << "in_dtype and out_dtype not equal, start converting..."
                << std::endl;
      auto in_kernel_type = framework::OpKernelType(in_dtype, place);
      auto out_kernel_type = framework::OpKernelType(out_dtype, place);
      framework::LoDTensor out;
      framework::TransDataType(in_kernel_type, out_kernel_type, tensor, &out);
      std::cout << "after the conversion, the dtype is "
                << static_cast<int>(framework::ToDataType(out.type()))
                << std::endl framework::SerializeToStream(fout, out, dev_ctx);
    } else {
      std::cout << "no conversion performed, the dtype is "
                << static_cast<int>(framework::ToDataType(tensor.type()))
                << std::endl framework::SerializeToStream(fout, tensor,
                                                          dev_ctx);
    }
  }
};

class SaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor ) Input tensor to be saved");
    AddComment(R"DOC(
Save operator

This operator will serialize and write a tensor variable to file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<int>("in_dtype",
                 "(int, default 5)"
                 "The data type of the input tensor")
        .SetDefault(static_cast<int>(framework::proto::VarType::FP32));
    AddAttr<int>("out_dtype",
                 "(int, default 5)"
                 "The data type of the converted tensor to be saved")
        .SetDefault(static_cast<int>(framework::proto::VarType::FP32));
    AddAttr<std::string>("file_path",
                         "(string)"
                         "The \"file_path\" where the variable will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save, ops::SaveOp, ops::SaveOpProtoMaker);

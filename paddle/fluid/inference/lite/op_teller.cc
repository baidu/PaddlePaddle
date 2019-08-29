// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/lite/op_teller.h"

namespace paddle {
namespace inference {
namespace lite {

// Just tell by the op_types.
struct SimpleOpTypeSetTeller : public Teller {
  SimpleOpTypeSetTeller() {
    teller_set.insert("leaky_relu");
    teller_set.insert("mul");
    teller_set.insert("elementwise_add");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& desc) override {
    return teller_set.count(op_type);
  }

 private:
  std::unordered_set<std::string> teller_set;
};

bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc) {
  for (auto& teller : tellers_) {
    if ((*teller)(op_type, desc)) return true;
  }
  return false;
}

OpTeller::OpTeller() { tellers_.emplace_back(new SimpleOpTypeSetTeller); }

}  // namespace lite
}  // namespace inference
}  // namespace paddle



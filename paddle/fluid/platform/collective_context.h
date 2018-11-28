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

#pragma once

#include <condition_variable>  // NOLINT
#include <string>
#include <vector>

namespace paddle {
namespace platform {

struct CollectiveContext {
  std::vector<std::string> endpoints_;
  int trainer_id_{0};

  std::string String() const {
    std::stringstream ss;
    ss << "endpoints_:";
    for (auto e : end_points_) {
      ss << e << ",";
    }

    ss << "trainer_id_:" << trainer_id_;

    return ss.str();
  }

  static CollectiveContext *GetInstance() {
    std::call_once(init_flag_,
                   [&]() { context_.reset(new CollectiveContext()); });
    return context_.get();
  }

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<CollectiveContext> context_;
};

}  // namespace platform
}  // namespace paddle

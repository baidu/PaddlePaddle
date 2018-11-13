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

#ifndef PADDLE_FLUID_FRAMEWORK_DATA_FEED_FACTORY_H_
#define PADDLE_FLUID_FRAMEWORK_DATA_FEED_FACTORY_H_

#include <string>
#include "paddle/framework/data_feed.h"

namespace paddle {
namespace framework {
class DataFeedFactory {
 public:
  static std::string DataFeedTypeList();
  static shared_ptr<DataFeed> CreateDataFeed(const char* data_feed_class);
};
}  // namespace framework
}  // namespace paddle

#endif  // PADDLE_FLUID_FRAMEWORK_DATA_FEED_FACTORY_H_

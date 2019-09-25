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

#include "paddle/fluid/framework/version.h"
#include <algorithm>
#include <sstream>

namespace paddle {
namespace framework {
bool IsProgramVersionSupported(int64_t version) {
  /* Not yet implemented. */
  return true;
}

bool IsTensorVersionSupported(uint32_t version) {
  static int num_supported =
      sizeof(kSupportedTensorVersion) / sizeof(kSupportedTensorVersion[0]);
  return std::find(kSupportedTensorVersion,
                   kSupportedTensorVersion + num_supported,
                   version) != kSupportedTensorVersion + num_supported;
}

std::string VersionString(const int64_t version) {
  std::stringstream buffer;
  const int major = version / MAJOR_COEFF;
  const int minor = (version - major * MAJOR_COEFF) / MINOR_COEFF;
  const int patch =
      (version - major * MAJOR_COEFF - minor * MINOR_COEFF) / PATCH_COEFF;
  buffer << major << "." << minor << "." << patch;
  return buffer.str();
}

}  // namespace framework
}  // namespace paddle

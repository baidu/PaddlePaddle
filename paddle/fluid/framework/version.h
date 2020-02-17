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

#pragma once

#include <cstdint>
#include <string>

namespace paddle {
namespace framework {

// Note:
// Program and Tensor that pass the IsXXXVersionSupported shold
// be supported by the current codes. Otherwise, it's a compatibility
// bug.

constexpr int MAJOR_COEFF = 1000000;
constexpr int MINOR_COEFF = 1000;
constexpr int PATCH_COEFF = 1;

// The program version the current codes generate.
#ifdef PADDLE_VERSION_INTEGER
constexpr int64_t kCurProgramVersion = PADDLE_VERSION_INTEGER;
#else
constexpr int64_t kCurProgramVersion = 0;
#endif

// The program version that was generated by previous or current codes
// and supported by current codes.
constexpr int64_t kSupportedProgramVersion[] = {0};

// Due to historical reasons, tensor version use uint32_t.
// The tensor version the current codes generate.
constexpr uint32_t kCurTensorVersion = 0;

// The tensor version that was generated by previous or current codes
// and supported by current codes.
constexpr uint32_t kSupportedTensorVersion[] = {0};

// WARNING: DO NOT use this interface, it may be discarded.
bool IsProgramVersionSupported(int64_t version);
// WARNING: DO NOT use this interface, it may be discarded.
bool IsTensorVersionSupported(uint32_t version);

std::string DumpVersion(const int64_t version);

}  // namespace framework
}  // namespace paddle

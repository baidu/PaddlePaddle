// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/buffered_allocator.h"
#include <algorithm>
#include <limits>
#include <utility>
#include "paddle/fluid/memory/allocation/allocation_with_underlying.h"

namespace paddle {
namespace memory {
namespace allocation {

BufferedAllocator::BufferedAllocator(std::shared_ptr<Allocator> allocator,
                                     const int64_t &threshold)
    : underlying_allocator_(std::move(allocator)) {
  PADDLE_ENFORCE_NOT_NULL(
      underlying_allocator_,
      "Underlying allocator of BufferedAllocator must be unmanaged");
  if (underlying_allocator_->IsAllocThreadSafe()) {
    mtx_.reset(new std::mutex());
  }
  if (threshold >= 0) {
    kMergeBufferThreshold = threshold;
  } else {
    kMergeBufferThreshold = 1 << 30;  // 1 GB
  }
}

BufferedAllocator::~BufferedAllocator() { FreeCache(-1UL); }

void BufferedAllocator::FreeCache(size_t size) {
  platform::LockGuardPtr<std::mutex> guard(mtx_);
  if (UNLIKELY(size == 0)) return;
  size_t cur = 0;
  while (!allocations_.empty()) {  // free the largest
    auto it = --allocations_.end();
    cur += it->second->size();
    delete it->second.release();
    allocations_.erase(it);
    if (cur >= size) return;
  }
}

bool BufferedAllocator::IsAllocThreadSafe() const {
  return this->underlying_allocator_->IsAllocThreadSafe();
}
void BufferedAllocator::Free(Allocation *allocation) {
  platform::LockGuardPtr<std::mutex> guard(mtx_);
  allocations_.emplace(allocation->size(), AllocationPtr(allocation));
  cache_size_.fetch_add(allocation->size());
}
Allocation *BufferedAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  {
    platform::LockGuardPtr<std::mutex> guard(mtx_);
    auto it = allocations_.lower_bound(size);
    // only use same size cache. Otherwise allocate a new one.
    if (it != allocations_.end()) {
      // AllocationWithUnderlying* result = it->second;
      AllocationPtr result(std::move(it->second));
      // AllocationPtr result(std::move(it->second));
      allocations_.erase(it);
      return static_cast<AllocationWithUnderlying *>(result.release());
    } else {
      if (cache_size_ > kMergeBufferThreshold) {
        FreeCache(std::numeric_limits<size_t>::max());
      }
    }
  }

  return new AllocationWithUnderlying(
      underlying_allocator_->Allocate(size, attr));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

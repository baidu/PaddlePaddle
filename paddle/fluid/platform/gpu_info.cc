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

#include "paddle/fluid/platform/gpu_info.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/string/split.h"

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
DECLARE_bool(enable_cublas_tensor_op_math);
DECLARE_string(selected_gpus);
DECLARE_uint64(gpu_memory_limit_mb);

constexpr static float fraction_reserve_gpu_memory = 0.05f;

namespace paddle {
namespace platform {

/* Here is a very simple CUDA “pro tip”: cudaDeviceGetAttribute() is a much
faster way to query device properties. You can see details in
https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
*/

inline std::string CudaErrorWebsite() {
  return "Please see detail in https://docs.nvidia.com/cuda/cuda-runtime-api"
         "/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c217824"
         "6db0a94a430e0038";
}

static int GetCUDADeviceCountImpl() {
  int driverVersion = 0;
  cudaError_t status = cudaDriverGetVersion(&driverVersion);

  if (!(status == cudaSuccess && driverVersion != 0)) {
    // No GPU driver
    return 0;
  }

  const auto *cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
  if (cuda_visible_devices != nullptr) {
    std::string cuda_visible_devices_str(cuda_visible_devices);
    if (std::all_of(cuda_visible_devices_str.begin(),
                    cuda_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "CUDA_VISIBLE_DEVICES is set to be empty. No GPU detected.";
      return 0;
    }
  }

  int count;
  auto error_code = cudaGetDeviceCount(&count);
  PADDLE_ENFORCE(
      error_code,
      "cudaGetDeviceCount failed in "
      "paddle::platform::GetCUDADeviceCountImpl, error code : %d, %s",
      error_code, CudaErrorWebsite());
  return count;
}

int GetCUDADeviceCount() {
  static auto dev_cnt = GetCUDADeviceCountImpl();
  return dev_cnt;
}

int GetCUDAComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int major, minor;

  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);
  PADDLE_ENFORCE_EQ(
      major_error_code, 0,
      "cudaDevAttrComputeCapabilityMajor failed in "
      "paddle::platform::GetCUDAComputeCapability, error code : %d, %s",
      major_error_code, CudaErrorWebsite());
  PADDLE_ENFORCE_EQ(
      minor_error_code, 0,
      "cudaDevAttrComputeCapabilityMinor failed in "
      "paddle::platform::GetCUDAComputeCapability, error code : %d, %s",
      minor_error_code, CudaErrorWebsite());
  return major * 10 + minor;
}

dim3 GetGpuMaxGridDimSize(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  dim3 ret;
  int size;
  auto error_code_x = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id);
  PADDLE_ENFORCE_EQ(error_code_x, 0,
                    "cudaDevAttrMaxGridDimX failed in "
                    "paddle::platform::GpuMaxGridDimSize, error code : %d, %s",
                    error_code_x, CudaErrorWebsite());
  ret.x = size;

  auto error_code_y = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id);
  PADDLE_ENFORCE_EQ(error_code_y, 0,
                    "cudaDevAttrMaxGridDimY failed in "
                    "paddle::platform::GpuMaxGridDimSize, error code : %d, %s",
                    error_code_y, CudaErrorWebsite());
  ret.y = size;

  auto error_code_z = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id);
  PADDLE_ENFORCE_EQ(error_code_z, 0,
                    "cudaDevAttrMaxGridDimZ failed in "
                    "paddle::platform::GpuMaxGridDimSize, error code : %d, %s",
                    error_code_z, CudaErrorWebsite());
  ret.z = size;
  return ret;
}

int GetCUDARuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int runtime_version = 0;
  auto error_code = cudaRuntimeGetVersion(&runtime_version);
  PADDLE_ENFORCE(error_code,
                 "cudaRuntimeGetVersion failed in "
                 "paddle::platform::GetCUDARuntimeVersion, error code : %d, %s",
                 error_code, CudaErrorWebsite());
  return runtime_version;
}

int GetCUDADriverVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int driver_version = 0;
  auto error_code = cudaDriverGetVersion(&driver_version);
  PADDLE_ENFORCE(error_code,
                 "cudaDriverGetVersion failed in "
                 "paddle::platform::GetCUDADriverVersion, error code : %d, %s",
                 error_code, CudaErrorWebsite());
  return driver_version;
}

bool TensorCoreAvailable() {
#if CUDA_VERSION >= 9000
  int device = GetCurrentDeviceId();
  int driver_version = GetCUDAComputeCapability(device);
  return driver_version >= 70;
#else
  return false;
#endif
}

int GetCUDAMultiProcessors(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  auto error_code =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id);
  PADDLE_ENFORCE(error_code,
                 "cudaDeviceGetAttribute failed in "
                 "paddle::platform::GetCUDAMultiProcess, error code : %d, %s",
                 error_code, CudaErrorWebsite());
  return count;
}

int GetCUDAMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  auto error_code = cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id);
  PADDLE_ENFORCE(
      error_code,
      "cudaDeviceGetAttribute failed in paddle::"
      "platform::GetCUDAMaxThreadsPerMultiProcessor, error code : %d, %s",
      error_code, CudaErrorWebsite());
  return count;
}

int GetCUDAMaxThreadsPerBlock(int id) {
  PADDLE_ENFORCE_LT(
      id, GetCUDADeviceCount(),
      platform::errors::InvalidArgument(
          "Device id must less than GPU count, but received id is:%d, "
          "GPU count is: %d.",
          id, GetCUDADeviceCount()));
  int count;
  auto error_code =
      cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id);
  PADDLE_ENFORCE_EQ(
      error_code, 0,
      platform::errors::InvalidArgument(
          "cudaDeviceGetAttribute returned error code should be 0, "
          "but received error code is: %d, %s",
          error_code, CudaErrorWebsite()));
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  auto error_code = cudaGetDevice(&device_id);
  PADDLE_ENFORCE(error_code,
                 "cudaGetDevice failed in "
                 "paddle::platform::GetCurrentDeviceId, error code : %d, %s",
                 error_code, CudaErrorWebsite());
  return device_id;
}

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices() {
  // use user specified GPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_gpus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_gpus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetCUDADeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void SetDeviceId(int id) {
  // TODO(qijun): find a better way to cache the cuda device count
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  auto error_code = cudaSetDevice(id);
  PADDLE_ENFORCE(error_code,
                 "cudaSetDevice failed in "
                 "paddle::platform::SetDeviced, error code : %d, %s",
                 error_code, CudaErrorWebsite());
}

void GpuMemoryUsage(size_t *available, size_t *total) {
  RecordedCudaMemGetInfo(available, total, platform::GetCurrentDeviceId());
}

size_t GpuAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  GpuMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_gpu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = GpuMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "GPU usage " << (available >> 20) << "M/" << (total >> 20)
           << "M, " << (available_to_alloc >> 20) << "M available to allocate";
  return available_to_alloc;
}

size_t GpuMaxAllocSize() {
  return std::max(GpuInitAllocSize(), GpuReallocSize());
}

static size_t GpuAllocSize(bool realloc) {
  size_t available_to_alloc = GpuAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(available_to_alloc, 0, "No enough available GPU memory");
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20 : available_to_alloc *
                                           FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_ENFORCE_GE(available_to_alloc, alloc_bytes,
                    "No enough available GPU memory");
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

size_t GpuInitAllocSize() { return GpuAllocSize(/* realloc = */ false); }

size_t GpuReallocSize() { return GpuAllocSize(/* realloc = */ true); }

size_t GpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

size_t GpuMaxChunkSize() {
  size_t max_chunk_size = GpuMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  auto error_code = cudaMemcpyAsync(dst, src, count, kind, stream);
  PADDLE_ENFORCE(error_code,
                 "cudaMemcpyAsync failed in paddle::platform::GpuMemcpyAsync "
                 "(%p -> %p, length: %d) error code : %d, %s",
                 src, dst, static_cast<int>(count), error_code,
                 CudaErrorWebsite());
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind) {
  auto error_code = cudaMemcpy(dst, src, count, kind);
  PADDLE_ENFORCE(error_code,
                 "cudaMemcpy failed in paddle::platform::GpuMemcpySync "
                 "(%p -> %p, length: %d) error code : %d, %s",
                 src, dst, static_cast<int>(count), error_code,
                 CudaErrorWebsite());
}

void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, cudaStream_t stream) {
  auto error_code =
      cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
  PADDLE_ENFORCE(
      error_code,
      "cudaMemcpyPeerAsync failed in paddle::platform::GpuMemcpyPeerAsync "
      "error code : %d, %s",
      error_code, CudaErrorWebsite());
}

void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count) {
  auto error_code = cudaMemcpyPeer(dst, dst_device, src, src_device, count);
  PADDLE_ENFORCE(error_code,
                 "cudaMemcpyPeer failed in paddle::platform::GpuMemcpyPeerSync "
                 "error code : %d, %s",
                 error_code, CudaErrorWebsite());
}

void GpuMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
  auto error_code = cudaMemsetAsync(dst, value, count, stream);
  PADDLE_ENFORCE(error_code,
                 "cudaMemsetAsync failed in paddle::platform::GpuMemsetAsync "
                 "error code : %d, %s",
                 error_code, CudaErrorWebsite());
}

void GpuStreamSync(cudaStream_t stream) {
  auto error_code = cudaStreamSynchronize(stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External(
          "cudaStreamSynchronize failed in paddle::platform::GpuStreamSync "
          "error code : %d, %s",
          error_code, CudaErrorWebsite()));
}

static void RaiseNonOutOfMemoryError(cudaError_t *status) {
  if (*status == cudaErrorMemoryAllocation) {
    *status = cudaSuccess;
  }

  PADDLE_ENFORCE_CUDA_SUCCESS(*status);

  *status = cudaGetLastError();
  if (*status == cudaErrorMemoryAllocation) {
    *status = cudaSuccess;
  }

  PADDLE_ENFORCE_CUDA_SUCCESS(*status);
}

class RecordedCudaMallocHelper {
 private:
  explicit RecordedCudaMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedCudaMallocHelper);

 public:
  static RecordedCudaMallocHelper *Instance(int dev_id) {
    std::call_once(once_flag_, [] {
      int dev_cnt = GetCUDADeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        instances_.emplace_back(
            new RecordedCudaMallocHelper(i, FLAGS_gpu_memory_limit_mb << 20));
      }
    });
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` gpu memory. Only cudaErrorMemoryAllocation
   * or cudaSuccess would be returned, and the cudaGetLastError() flag
   * would be clear.
   */
  cudaError_t Malloc(void **ptr, size_t size) {
    LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_ + size > limit_size_)) {
      return cudaErrorMemoryAllocation;
    }

    CUDADeviceGuard guard(dev_id_);
    auto result = cudaMalloc(ptr, size);
    if (result == cudaSuccess) {
      if (NeedRecord()) {
        cur_size_ += size;
      }
      return cudaSuccess;
    } else {
      RaiseNonOutOfMemoryError(&result);
      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError. Therefore, we can
      // return cudaErrorMemoryAllocation directly here.
      return cudaErrorMemoryAllocation;
    }
  }

  /**
   * Free gpu memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void *ptr, size_t size) {
    // Purposefully allow cudaErrorCudartUnloading, because
    // that is returned if you ever call cudaFree after the
    // driver has already shutdown. This happens only if the
    // process is terminating, in which case we don't care if
    // cudaFree succeeds.
    CUDADeviceGuard guard(dev_id_);
    auto err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          err, platform::errors::External("cudaFree raises unexpected error"));
      if (NeedRecord()) {
        std::lock_guard<std::mutex> guard(*mtx_);
        cur_size_ -= size;
      }
    } else {
      cudaGetLastError();  // clear the error flag when cudaErrorCudartUnloading
    }
  }

  void GetMemInfo(size_t *avail, size_t *total) {
    {
      CUDADeviceGuard guard(dev_id_);
      auto result = cudaMemGetInfo(avail, total);
      if (result != cudaSuccess) {
        *avail = 0;
      }
      RaiseNonOutOfMemoryError(&result);
    }

    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      *avail = std::min(*avail, limit_size_ - cur_size_);
      *total = std::min(*total, limit_size_);
    }
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

 private:
  const int dev_id_;
  const uint64_t limit_size_;
  uint64_t cur_size_{0};

  std::unique_ptr<std::mutex> mtx_;

  static std::once_flag once_flag_;
  static std::vector<std::unique_ptr<RecordedCudaMallocHelper>> instances_;
};

std::once_flag RecordedCudaMallocHelper::once_flag_;
std::vector<std::unique_ptr<RecordedCudaMallocHelper>>
    RecordedCudaMallocHelper::instances_;

cudaError_t RecordedCudaMalloc(void **ptr, size_t size, int dev_id) {
  return RecordedCudaMallocHelper::Instance(dev_id)->Malloc(ptr, size);
}

void RecordedCudaFree(void *p, size_t size, int dev_id) {
  return RecordedCudaMallocHelper::Instance(dev_id)->Free(p, size);
}

void RecordedCudaMemGetInfo(size_t *avail, size_t *total, int dev_id) {
  RecordedCudaMallocHelper::Instance(dev_id)->GetMemInfo(avail, total);
}

}  // namespace platform
}  // namespace paddle

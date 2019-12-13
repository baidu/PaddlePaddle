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

#ifndef _WIN32
#pragma once

#include <stdio.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#define NCCL_ID_VARNAME "NCCLID"

namespace paddle {
namespace platform {

inline ncclDataType_t ToNCCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return ncclFloat;
  } else if (type == framework::proto::VarType::FP64) {
    return ncclDouble;
  } else if (type == framework::proto::VarType::INT32) {
    return ncclInt;
  } else if (type == framework::proto::VarType::INT64) {
    return ncclInt64;
  } else if (type == framework::proto::VarType::FP16) {
    return ncclFloat16;
  } else {
    PADDLE_THROW("Not supported");
  }
}

// NOTE(minqiyang): according to the ncclGroupEnd documentations:
// https://docs.nvidia.com/deeplearning/sdk/nccl-api/ncclapidoc.html,
// ncclGroupEnd will wait for all communicators to be initialized, which will
// cause blocking problem when a runtime_error was thrown, so try only guard
// NCCL actions when use it.
class NCCLGroupGuard {
 public:
  static std::mutex &NCCLMutex() {
    static std::mutex mtx;
    return mtx;
  }

  inline NCCLGroupGuard() {
    NCCLMutex().lock();
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupStart());
  }

  inline ~NCCLGroupGuard() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupEnd());
    NCCLMutex().unlock();
  }
};

// The communication unit holding a connected ncclComm_t and an independent
// cudaStream_t. The lifetime of the ncclComm_t is maintained by this unit,
// so we should not destroy the ncclComm_t explicitly.
class NCCLContext {
 public:
  NCCLContext(int dev_id, ncclComm_t comm)
      : dev_ctx_(new CUDADeviceContext(CUDAPlace(dev_id))) {
    dev_ctx_->set_nccl_comm(comm);
  }
  virtual ~NCCLContext() {}

  cudaStream_t stream() const { return dev_ctx_->stream(); }

  ncclComm_t comm() const { return dev_ctx_->nccl_comm(); }

  int device_id() const {
    return boost::get<platform::CUDAPlace>(dev_ctx_->GetPlace()).device;
  }

  CUDADeviceContext *dev_ctx() const { return dev_ctx_.get(); }

 protected:
  // CUDADeviceContext is the real container of the ncclComm_t, and it destroys
  // ncclComm_t in its destructor
  // TODO(liuyi05): try to move the destruction of the ncclComm_t out of
  // CUDADeviceContext and keep it in the same class with its construction
  std::unique_ptr<CUDADeviceContext> dev_ctx_;

  DISABLE_COPY_AND_ASSIGN(NCCLContext);
};

// A container preserving the mapping from device id to NCCLContext which has
// a corresponding level to "ring" in NCCL communication. Users retrieve the
// NCCLContext instance from it by the device id of a given CUDAPlace.
//
// The status of this container may be diverse in different training modes:
//
// 1. Multiprocess (with one thread each) mode:
//    The minimal training unit is a process in this mode, and each process,
//    taking up one device, has a rank id in a communication ring. The
//    NCCLContextMap holds the ring information and contains only one
//    NCCLContext.
//
// 2. One process with multithread mode:
//    In this mode, the minimal training unit is a thread, and all threads are
//    working in one process. Within a process, we can initialize all
//    NCCLContext instances once in a ring and put them all to a single
//    NCCLContextMap.
//
// 3. Multiprocess with multithread mode:
//    This mode is in the case of training with multimachine. The ring consists
//    of all threads across processes over machines. The NCCLContext is created
//    by each thread, taking up one device, and all NCCLContext instances in
//    one process preserved in a single NCCLContextMap.
//
// As one training procedure may use several rings to communicate, there could
// be several NCCLContextMaps.
class NCCLContextMap {
 public:
  explicit NCCLContextMap(const std::vector<platform::Place> &places,
                          ncclUniqueId *nccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0);

  const std::map<int, std::unique_ptr<NCCLContext>> &contexts() const {
    return contexts_;
  }

  CUDADeviceContext *DevCtx(int dev_id) const { return at(dev_id)->dev_ctx(); }

  CUDADeviceContext *DevCtx(platform::Place p) const {
    return DevCtx(boost::get<CUDAPlace>(p).device);
  }

  NCCLContext *at(platform::Place p) const {
    return at(boost::get<CUDAPlace>(p).device);
  }

  NCCLContext *at(int dev_id) const { return contexts_.at(dev_id).get(); }

  size_t size() const { return contexts_.size(); }

  size_t count(int dev_id) const { return contexts_.count(dev_id); }

  NCCLContext *GetSingle() const {
    PADDLE_ENFORCE_EQ(contexts_.size(), 1);
    return contexts_.begin()->second.get();
  }

  void WaitAll() {
    for (auto &p : contexts_) {
      p.second->dev_ctx()->Wait();
    }
  }

 private:
  std::map<int, std::unique_ptr<NCCLContext>> contexts_;
  std::vector<int> order_;

  ncclUniqueId *nccl_id_;
  int ntrainers_;
  int trainer_id_;

  DISABLE_COPY_AND_ASSIGN(NCCLContextMap);
};

// A singleton of NCCL communication context. It creates NCCLContext in
// communication rings and maintains lifetime of all NCCL resources.
// Each ring has an unique ID, which could be used to retrieve
// the NCCLContextMap. So if both ring index and device id are given, the
// NCCLContext instance could be determined.
//
// For compatibility, we preserve the range [0, 1000) of ids for default
// rings and users could use ids beyond 1000.
//
// E.g. for a hierarchical communication case,
//
//    11 - 12   21 - 22
//     |    |    |    |
//    13 - 14 - 23 - 24
//          |    |
//    31 - 32 - 41 - 42
//     |    |    |    |
//    33 - 34   43 - 44
//
// we create (14,23,32,41) as the top ring, and (11,12,13,14), (21,22,23,24),
// (31,32,33,34), (41,42,43,44) as bottoms rings respectively.
//
// We could also use a single communication ring for the flatten case
class NCCLReference {
 public:
  using ContextMapVector = std::vector<std::unique_ptr<NCCLContextMap>>;

  // for compatible
  friend class NCCLCommunicator;

  static NCCLReference &Instance() {
    static NCCLReference instance;
    return instance;
  }

  NCCLReference() = default;
  ~NCCLReference() = default;

  void InitFlattenRing(const std::vector<Place> &places,
                       const std::vector<std::string> &endpoints,
                       size_t trainer_id, size_t nrings = 1);

  void Init2DRing(const std::vector<Place> &places,
                  const std::vector<std::string> &endpoints, size_t trainer_id,
                  size_t nrings = 1);

  // @Deprecated. Only for compatible
  void InitFlattenRing(const std::vector<Place> &places,
                       const std::vector<ncclUniqueId *> &nccl_ids,
                       size_t trainer_num, size_t trainer_id);

  // @Deprecated. Only for compatible to suppose multi-process mode
  void InitHierarchicalRing(const std::vector<Place> &places,
                            const std::vector<ncclUniqueId *> &inter_nccl_ids,
                            const std::vector<ncclUniqueId *> &exter_nccl_ids,
                            size_t nranks, size_t rank_id, size_t inter_ranks);

  // TODO(liuyi05): put these three InitNCCLxxx functions to private section.
  // c_comm_init_xxx operators need these functions by now, we will solve it
  // later.
  void InitNCCLContexts(const std::vector<Place> &places,
                        ncclUniqueId *nccl_id = nullptr, size_t ntrainers = 1,
                        size_t trainer_id = 0,
                        ContextMapVector *rings = nullptr);

  void InitAllNCCLContexts(const std::vector<Place> &places,
                           ContextMapVector *rings = nullptr) {
    InitNCCLContexts(places, nullptr, 1, 0);
  }

  void InitNCCLContext(ncclUniqueId *nccl_id, size_t nranks, size_t rank,
                       Place place) {
    InitNCCLContexts({place}, nccl_id, nranks, rank);
  }

  void AllReduce(const void *send, void *recv, size_t count,
                 ncclDataType_t datatype, ncclRedOp_t op, const Place &place,
                 cudaStream_t stream = nullptr, size_t order = 0);

  void AllReduce2D(const void *send, void *recv, size_t count,
                   ncclDataType_t datatype, ncclRedOp_t op, const Place &place,
                   cudaStream_t stream = nullptr, size_t order = 0);

  NCCLContextMap *GetContextMap(int index) const {
    return flat_rings_.at(index).get();
  }

  NCCLContextMap *GetDefaultContextMap() const { return GetContextMap(0); }

  // retrieve a communicator by the ring index
  NCCLContext *Get(size_t index) const {
    PADDLE_ENFORCE_LT(index, flat_rings_.size());
    return flat_rings_.at(index)->GetSingle();
  }

  // retrieve a communicator by the ring index and the device id
  NCCLContext *Get(int index, int dev_id) const {
    PADDLE_ENFORCE_LT(index, flat_rings_.size());
    PADDLE_ENFORCE_GT(
        flat_rings_.at(index)->count(dev_id), 0,
        "comunicator at device id %d has not been initialized in ring %d",
        dev_id, index);
    return flat_rings_.at(index)->at(dev_id);
  }

  // retrieve a communicator by the ring index and place
  NCCLContext *Get(int index, Place place) const {
    return Get(index, boost::get<CUDAPlace>(place).device);
  }

 protected:
  static const char s_nccl_id_var_name_[];

  std::once_flag once_flag_;

  // for ring-based allreduce
  std::vector<std::unique_ptr<NCCLContextMap>> flat_rings_;

  // for 2D allreduce
  std::vector<std::unique_ptr<NCCLContextMap>> d2_inter_rings_;
  std::vector<std::unique_ptr<NCCLContextMap>> d2_exter_rings_;
  size_t d2_inter_ranks_;
  size_t d2_exter_ranks_;

  // for hierarchical allreduce
  std::vector<std::unique_ptr<NCCLContextMap>> h_inter_rings_;
  std::vector<std::unique_ptr<NCCLContextMap>> h_exter_rings_;
  size_t h_inter_ranks_;
  size_t h_exter_ranks_;

  void ReleaseNCCLResource();

  void GenerateAndSend(framework::Scope *scope,
                       const std::vector<std::string> &endpoints);

  void GetIdByServer(framework::Scope *scope,
                     const std::vector<std::string> &endpoints);

  void AssignNCCLId(const std::vector<std::string> &endpoints,
                    size_t trainer_id, ncclUniqueId *nccl_id);

  //// just used for sync_batch_norm op.
  // std::unique_ptr<NCCLContextMap> sync_batch_norm_ctx_;

  DISABLE_COPY_AND_ASSIGN(NCCLReference);
};

// TODO(liuyi05): need to check these global functions
inline std::string GetFlatNCCLVarName(size_t pos) {
  if (pos == 0) {
    return NCCL_ID_VARNAME;
  }
  return string::Sprintf("%s_%d", NCCL_ID_VARNAME, static_cast<int>(pos));
}

inline std::string GetHierarchicalExterNCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_exter_%s_%d", NCCL_ID_VARNAME,
                         static_cast<int>(pos));
}
inline std::string GetHierarchicalInterNCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_inter_%s_%d", NCCL_ID_VARNAME,
                         static_cast<int>(pos));
}

class NCCLCommunicator {
 public:
  NCCLCommunicator() {}
  virtual ~NCCLCommunicator() PADDLE_MAY_THROW {}
  NCCLContextMap *DefaultFlatCtx() const {
    auto &flat_rings = NCCLReference::Instance().flat_rings_;
    if (flat_rings.size() == 0) {
      return nullptr;
    }

    return flat_rings[0].get();
  }

  const std::vector<std::unique_ptr<NCCLContextMap>> *GetFlatCtxs() {
    return &NCCLReference::Instance().flat_rings_;
  }

  NCCLContextMap *GetFlatCtx(size_t run_order) const {
    auto &flat_rings = NCCLReference::Instance().flat_rings_;
    return flat_rings[run_order % flat_rings.size()].get();
  }

  NCCLContextMap *GetRunEnvNCCLCtx(size_t run_order,
                                   bool use_hierarchical_allreduce) const {
    if (!use_hierarchical_allreduce) {
      return GetFlatCtx(run_order);
    }

    return GetHierarchicalInterCtx(run_order);
  }

  // TODO(liuyi05)
  /*
   *When nccl inits nccl comm using ncclCommInitAll, it meets error when
   *allreduce ophandle and sync_batch_norm_op use ncclallreduce parallelly. So
   *create a new nccl comm for sync_batch_norm_op. And these codes should be
   *polished with a unified nccl management.
  */
  NCCLContextMap *GetSyncBatchNormCtx(
      framework::Scope *scope, const std::vector<platform::Place> &places) {
    auto *nccl_id_var = scope->FindVar(NCCL_ID_VARNAME);
    if (nccl_id_var != nullptr) {
      return DefaultFlatCtx();
    }

    if (sync_batch_norm_ctx_.get() == nullptr) {
      sync_batch_norm_ctx_.reset(new NCCLContextMap(places));
    }
    return sync_batch_norm_ctx_.get();
  }

  void InitFlatCtxs(const std::vector<platform::Place> &places,
                    const std::vector<ncclUniqueId *> &nccl_ids,
                    size_t trainers_num, size_t trainer_id) {
    auto &ins = NCCLReference::Instance();
    ins.InitFlattenRing(places, nccl_ids, trainers_num, trainer_id);
  }

  void InitHierarchicalCtxs(const std::vector<platform::Place> &places,
                            const std::vector<ncclUniqueId *> &inter_nccl_ids,
                            const std::vector<ncclUniqueId *> &exter_nccl_ids,
                            size_t trainers_num, size_t trainer_id,
                            size_t inter_trainers_num,
                            size_t exter_trainers_num) {
    PADDLE_ENFORCE_EQ(trainers_num, inter_trainers_num * exter_trainers_num,
                      "trainers_num:%llu != inter_trainers_num:%llu * "
                      "exter_trainers_num:%llu",
                      trainers_num, inter_trainers_num, exter_trainers_num);
    PADDLE_ENFORCE_GT(inter_trainers_num, 1, "inter_trainers_num:%llu must > 1",
                      inter_trainers_num);
    auto &ins = NCCLReference::Instance();
    ins.InitHierarchicalRing(places, inter_nccl_ids, exter_nccl_ids,
                             trainers_num, trainer_id, inter_trainers_num);
  }

  bool NeedExterAllReduce() const {
    return NCCLReference::Instance().h_exter_rings_.size() > 0;
  }

  NCCLContextMap *GetHierarchicalInterCtx(size_t run_order) const {
    auto &inter_rings = NCCLReference::Instance().h_inter_rings_;
    PADDLE_ENFORCE(inter_rings.size() > 0,
                   "must init hierarchical ctxs first!");
    return inter_rings[run_order % inter_rings.size()].get();
  }

  NCCLContextMap *GetHierarchicalExterCtx(size_t run_order) const {
    auto &exter_rings = NCCLReference::Instance().h_exter_rings_;
    PADDLE_ENFORCE(exter_rings.size() > 0,
                   "must init hierarchical ctxs first!");
    return exter_rings[run_order % exter_rings.size()].get();
  }

  const std::vector<std::unique_ptr<NCCLContextMap>>
      *GetHierarchicalInterCtxs() {
    return &NCCLReference::Instance().h_inter_rings_;
  }

  const std::vector<std::unique_ptr<NCCLContextMap>>
      *GetHierarchicalExterCtxs() {
    return &NCCLReference::Instance().h_exter_rings_;
  }

 protected:
  // just used for sync_batch_norm op.
  // Is it neccessary to use another stream to sync batch norm?
  std::unique_ptr<NCCLContextMap> sync_batch_norm_ctx_;
};

}  // namespace platform
}  // namespace paddle
#endif

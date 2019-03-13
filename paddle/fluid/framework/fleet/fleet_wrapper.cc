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

#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include <utility>
#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;
std::shared_ptr<FleetWrapper> FleetWrapper::s_instance_ = NULL;
bool FleetWrapper::is_initialized_ = false;

#ifdef PADDLE_WITH_PSLIB
template <class AR>
paddle::ps::Archive<AR>& operator<<(paddle::ps::Archive<AR>& ar,
                                    const MultiSlotType& ins) {
  ar << ins.GetType();
  ar << ins.GetOffset();
  ar << ins.GetFloatData();
  ar << ins.GetUint64Data();
  return ar;
}

template <class AR>
paddle::ps::Archive<AR>& operator>>(paddle::ps::Archive<AR>& ar,
                                    MultiSlotType& ins) {
  ar >> ins.MutableType();
  ar >> ins.MutableOffset();
  ar >> ins.MutableFloatData();
  ar >> ins.MutableUint64Data();
  return ar;
}
#endif

#ifdef PADDLE_WITH_PSLIB
std::shared_ptr<paddle::distributed::PSlib> FleetWrapper::pslib_ptr_ = NULL;
#endif

void FleetWrapper::InitServer(const std::string& dist_desc, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init server";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::PSlib>(
        new paddle::distributed::PSlib());
    pslib_ptr_->init_server(dist_desc, index);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
#endif
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<uint64_t>& host_sign_list,
                              int node_num, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init worker";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::PSlib>(
        new paddle::distributed::PSlib());
    pslib_ptr_->init_worker(dist_desc,
                            const_cast<uint64_t*>(host_sign_list.data()),
                            node_num, index);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Worker can be initialized only once";
  }
#endif
}

void FleetWrapper::StopServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to stop server";
  pslib_ptr_->stop_server();
#endif
}

uint64_t FleetWrapper::RunServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server";
  return pslib_ptr_->run_server();
#else
  return 0;
#endif
}

void FleetWrapper::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                 int node_num) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to gather server ips";
  pslib_ptr_->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                             node_num);
#endif
}

void FleetWrapper::PullSparseVarsSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<::std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (auto name : var_names) {
    Variable* var = scope.FindVar(name);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
    fea_values->resize(fea_keys->size() + 1);
    for (auto& t : *fea_values) {
      t.resize(fea_value_dim);
    }
    std::vector<float*> pull_result_ptr;
    for (auto& t : *fea_values) {
      pull_result_ptr.push_back(t.data());
    }
    auto status = pslib_ptr_->_worker_ptr->pull_sparse(
        pull_result_ptr.data(), table_id, fea_keys->data(), fea_keys->size());
    pull_sparse_status.push_back(std::move(status));
  }
  for (auto& t : pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      exit(-1);
    }
  }
#endif
}

void FleetWrapper::PullDenseVarsAsync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* pull_dense_status) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  regions.resize(var_names.size());
  for (auto i = 0u; i < var_names.size(); ++i) {
    Variable* var = scope.FindVar(var_names[i]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions[i] = std::move(reg);
  }
  auto status =
      pslib_ptr_->_worker_ptr->pull_dense(regions.data(), regions.size(), tid);
  pull_dense_status->push_back(std::move(status));
#endif
}

void FleetWrapper::PullDenseVarsSync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  regions.reserve(var_names.size());
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions.emplace_back(std::move(reg));
  }
  auto status =
      pslib_ptr_->_worker_ptr->pull_dense(regions.data(), regions.size(), tid);
  status.wait();
#endif
}

void FleetWrapper::PushDenseVarsSync(
    Scope* scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {}

void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* push_sparse_status) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int count = tensor->numel();
    float* g = tensor->data<float>();
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }
  auto status = pslib_ptr_->_worker_ptr->push_dense(regions.data(),
                                                    regions.size(), table_id);
  push_sparse_status->push_back(std::move(status));
#endif
}

void FleetWrapper::PushSparseVarsWithLabelAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<uint64_t>& fea_keys, const std::vector<float>& fea_labels,
    const std::vector<std::string>& sparse_key_names,
    const std::vector<std::string>& sparse_grad_names, const int emb_dim,
    std::vector<std::vector<float>>* push_values,
    std::vector<::std::future<int32_t>>* push_sparse_status) {
#ifdef PADDLE_WITH_PSLIB
  int offset = 2;
  uint64_t fea_idx = 0u;
  for (size_t i = 0; i < sparse_key_names.size(); ++i) {
    LOG(WARNING) << "sparse key names[" << i << "]: " << sparse_key_names[i];
    LOG(WARNING) << "sparse grad names[" << i << "]: " << sparse_grad_names[i];
    Variable* g_var = scope.FindVar(sparse_grad_names[i]);
    CHECK(g_var != nullptr) << "var[" << sparse_grad_names[i] << "] not found";
    LoDTensor* g_tensor = g_var->GetMutable<LoDTensor>();
    if (g_tensor == NULL) {
      LOG(ERROR) << "var[" << sparse_key_names[i] << "] not found";
      exit(-1);
    }
    float* g = g_tensor->data<float>();
    Variable* var = scope.FindVar(sparse_key_names[i]);
    CHECK(var != nullptr) << "var[" << sparse_key_names[i] << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (tensor == NULL) {
      LOG(ERROR) << "var[" << sparse_key_names[i] << "] not found";
      exit(-1);
    }
    int len = tensor->numel();
    LOG(WARNING) << " tensor len: " << len;
    int64_t* ids = tensor->data<int64_t>();
    push_values->resize(fea_keys.size() + 1);
    for (auto& t : *push_values) {
      t.resize(emb_dim + offset);
    }

    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        g += emb_dim;
        continue;
      }
      LOG(WARNING) << "going to memcpy";
      memcpy((*push_values)[fea_idx].data() + offset, g,
             sizeof(float) * emb_dim);
      LOG(WARNING) << "show";
      (*push_values)[fea_idx][0] = 1.0f;
      LOG(WARNING) << "click";
      (*push_values)[fea_idx][1] = static_cast<float>(fea_labels[fea_idx]);
      LOG(WARNING) << "offset";
      g += emb_dim;
      fea_idx++;
    }
  }
  CHECK(fea_idx == fea_keys.size()) << "fea_idx: " << fea_idx
                                    << "features size: " << fea_keys.size();
  std::vector<float*> push_g_vec;
  for (auto i = 0u; i < fea_keys.size(); ++i) {
    push_g_vec.push_back((*push_values)[i].data());
  }
  auto status = pslib_ptr_->_worker_ptr->push_sparse(
      table_id, fea_keys.data(), (const float**)push_g_vec.data(),
      fea_keys.size());
  push_sparse_status->push_back(std::move(status));

#endif
}

int FleetWrapper::RegisterClientToClientMsgHandler(
    int msg_type, MsgHandlerFunc handler) {
#ifdef PADDLE_WITH_PSLIB
  pslib_ptr_->_worker_ptr->registe_client2client_msg_handler(
      msg_type, handler);
#else
  VLOG(0) << "FleetWrapper::RegisterClientToClientMsgHandler"
          << " does nothing when no pslib";
#endif
  return 0;
}

int FleetWrapper::SendClientToClientMsg(
    int msg_type, int to_client_id, const std::string& msg) {
#ifdef PADDLE_WITH_PSLIB
  pslib_ptr_->_worker_ptr->send_client2client_msg(
      msg_type, to_client_id, msg);
#else
  VLOG(0) << "FleetWrapper::SendClientToClientMsg"
          << " does nothing when no pslib";
#endif
  return 0;
}

std::default_random_engine& FleetWrapper::LocalRandomEngine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
    engine_wrapper_t() {
      struct timespec tp;
      clock_gettime(CLOCK_REALTIME, &tp);
      double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
      static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
      engine.seed(sseq);
    }
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

template <typename T>
void FleetWrapper::Serialize(const T& t, std::string* str) {
#ifdef PADDLE_WITH_PSLIB
  paddle::ps::BinaryArchive ar;
  ar << t;
  *str = std::string(ar.buffer(), ar.length());
#else
  VLOG(0) << "FleetWrapper::Serialize does nothing when no pslib";
#endif
}

template <typename T>
void FleetWrapper::Deserialize(T* t, const std::string& str) {
#ifdef PADDLE_WITH_PSLIB
  paddle::ps::BinaryArchive ar;
  ar.set_read_buffer(const_cast<char*>(str.c_str()), str.length(), nullptr);
  *t = ar.get<T>();
#else
  VLOG(0) << "FleetWrapper::Deserialize does nothing when no pslib";
#endif
}

template void FleetWrapper::Serialize<std::vector<MultiSlotType>>(
    const std::vector<MultiSlotType>&, std::string*);
template void FleetWrapper::Deserialize(std::vector<MultiSlotType>*,
                                        const std::string&);

}  // end namespace framework
}  // end namespace paddle

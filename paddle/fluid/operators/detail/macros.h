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

#pragma once

#ifdef PADDLE_WITH_DISTRIBUTE

#ifdef PADDLE_WITH_GRPC

#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/detail/grpc_server.h"
#define RPCSERVER_T paddle::operators::detail::AsyncGRPCServer
#define RPCCLIENT_T paddle::operators::detail::GRPCClient

#else  // PADDLE_WITH_GRPC

#include "paddle/fluid/operators/detail/brpc_client.h"
#include "paddle/fluid/operators/detail/brpc_server.h"
#define RPCSERVER_T paddle::operators::detail::AsyncBRPCServer
#define RPCCLIENT_T paddle::operators::detail::BRPCClient

#endif  // PADDLE_WITH_GRPC

#endif  // PADDLE_WITH_DISTRIBUTE

#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collective_runtime
import parameter_server_runtime


class RuntimeFactory(object):
    def _create_runtime(final_dist_strategy, role_maker, opt_ops, params_grads):
        if role_maker._is_collective:
            return collective_runtime.CollectiveRuntime(
                final_dist_strategy, role_maker, opt_ops, params_grads)
        else:
            return parameter_server_runtime.PSRuntime(
                final_dist_strategy, role_maker, opt_ops, params_grads)

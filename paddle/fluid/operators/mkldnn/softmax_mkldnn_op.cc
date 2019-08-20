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

#include <iostream>
#include "mkldnn.hpp"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using mkldnn::memory;  // Note: paddle has also "memory" namespace
using mkldnn::primitive;
using mkldnn::prop_kind;
using mkldnn::softmax_backward;
using mkldnn::softmax_forward;
using mkldnn::stream;
using platform::to_void_cast;

class SoftmaxMKLDNNHandler : public platform::MKLDNNHandler {
 public:
  SoftmaxMKLDNNHandler(const platform::MKLDNNDeviceContext& dev_ctx,
                       mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {}

  SoftmaxMKLDNNHandler(
      std::shared_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd,
      std::shared_ptr<mkldnn::softmax_backward::primitive_desc> softmax_bwd_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        softmax_pd_(softmax_pd),
        softmax_bwd_pd_(softmax_bwd_pd) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    key_ += "-BWD";
  }

  std::shared_ptr<softmax_forward::primitive_desc>
  AcquireSoftmaxPrimitiveDescriptor(const softmax_forward::desc& softmax_desc,
                                    const mkldnn::engine& engine) {
    // Softmax PD has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    const std::string key_softmax_pd = key_common_ + "@softmax_pd";

    softmax_pd_ = std::static_pointer_cast<softmax_forward::primitive_desc>(
        dev_ctx_.GetBlob(key_softmax_pd));
    if (softmax_pd_ == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      softmax_pd_ = std::static_pointer_cast<softmax_forward::primitive_desc>(
          dev_ctx_.GetBlob(key_softmax_pd));
      if (softmax_pd_ == nullptr) {
        softmax_pd_.reset(
            new softmax_forward::primitive_desc(softmax_desc, engine));
        dev_ctx_.SetBlob(key_softmax_pd, softmax_pd_);
      }
    }

    return softmax_pd_;
  }

  std::shared_ptr<mkldnn::softmax_forward> AcquireSoftmax() {
    /*Generate key*/
    auto prim_key = key_ + "@softmax_p";

    auto softmax_p = std::static_pointer_cast<mkldnn::softmax_forward>(
        dev_ctx_.GetBlob(prim_key));
    if (softmax_p == nullptr) {
      softmax_p = std::make_shared<mkldnn::softmax_forward>(*softmax_pd_);
      dev_ctx_.SetBlob(prim_key, softmax_p);
    }

    return softmax_p;
  }

  std::shared_ptr<mkldnn::softmax_backward> AcquireSoftmaxBackward() {
    auto prim_key = key_ + "@softmax_bwd_p";
    auto softmax_bwd_p = std::static_pointer_cast<mkldnn::softmax_backward>(
        dev_ctx_.GetBlob(prim_key));
    if (softmax_bwd_p == nullptr) {
      softmax_bwd_p =
          std::make_shared<mkldnn::softmax_backward>(*softmax_bwd_pd_);
      dev_ctx_.SetBlob(prim_key, softmax_bwd_p);
    }

    return softmax_bwd_p;
  }

 private:
  std::shared_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd_;
  std::shared_ptr<mkldnn::softmax_backward::primitive_desc> softmax_bwd_pd_;
};

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();
    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE_EQ(
        input->dims(), output->dims(),
        "The shape of softmax's input and output must be identical.");

    // make sure 'output' holds memory, which will be shared by
    // 'flattened_output' later.
    output->mutable_data<T>(ctx.GetPlace());

    // flatten input and output to 2-D matrixs
    auto dims = input->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);
    framework::Tensor flattened_input;
    framework::Tensor flattened_output;
    flattened_input.ShareDataWith(*input).Resize(flattened_dims);
    flattened_output.ShareDataWith(*output).Resize(flattened_dims);

    const T* input_data = flattened_input.data<T>();
    T* output_data = flattened_output.mutable_data<T>(ctx.GetPlace());

    std::vector<int64_t> src_tz = paddle::framework::vectorize(flattened_dims);
    std::vector<int64_t> dst_tz = src_tz;
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};
    // Generate keys for storing/retriving primitives for this operator
    const std::string key =
        platform::MKLDNNHandler::GetHash(softmax_tz, ctx.op().Output("Out"));

    SoftmaxMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);
    // Currently only NC data format is supported
    auto softmax_md = MKLDNNMemDesc(
        {softmax_tz}, platform::MKLDNNGetDataType<T>(), memory::format_tag::nc);
    // Normalization is made after innermost dimension eg. C out of NC
    auto softmax_desc = softmax_forward::desc(prop_kind::forward_scoring,
                                              softmax_md, 1 /*dim: C*/);

    auto softmax_pd =
        handler.AcquireSoftmaxPrimitiveDescriptor(softmax_desc, mkldnn_engine);

    auto softmax_src_memory_p =
        handler.AcquireSrcMemory(softmax_md, to_void_cast<T>(input_data));
    auto softmax_dst_memory_p =
        handler.AcquireDstMemory(softmax_md, to_void_cast<T>(output_data));
    auto softmax_p = handler.AcquireSoftmax();

    mkldnn::stream astream(mkldnn_engine);
    softmax_p->execute(astream, {{MKLDNN_ARG_SRC, *softmax_src_memory_p},
                                 {MKLDNN_ARG_DST, *softmax_dst_memory_p}});
    astream.wait();

    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T threshold = exp(-64);
      for (int i = 0; i < dst_tz[0] * dst_tz[1]; ++i) {
        output_data[i] =
            output_data[i] < threshold ? threshold : output_data[i];
      }
    }
  }
};

template <typename T>
class SoftmaxMKLDNNGradKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();
    const Tensor* output = ctx.Input<Tensor>("Out");
    auto* dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
    auto* dx =
        ctx.template Output<framework::Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        dout->dims(), dx->dims(),
        "The shape of softmax_grad's input and output must be identical.");

    // make sure 'dx' holds memory, which will be shared by 'flattened_dx'
    // later.
    dx->template mutable_data<T>(ctx.GetPlace());

    auto dims = dout->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);
    framework::Tensor flattened_output;
    framework::Tensor flattened_dout;
    framework::Tensor flattened_dx;
    flattened_output.ShareDataWith(*output).Resize(flattened_dims);
    flattened_dout.ShareDataWith(*dout).Resize(flattened_dims);
    flattened_dx.ShareDataWith(*dx).Resize(flattened_dims);

    const T* dst_data = flattened_output.data<T>();
    const T* diff_dst_ptr = flattened_dout.template data<T>();
    T* diff_src_ptr = flattened_dx.template mutable_data<T>(ctx.GetPlace());

    std::vector<int64_t> dst_tz = paddle::framework::vectorize(flattened_dims);
    std::vector<int64_t> src_tz(dst_tz);

    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};
    // Currently only supports NC data format
    // retrieve eltwise primitive desc from device context
    const std::string key =
        platform::MKLDNNHandler::GetHash(softmax_tz, ctx.op().Input("Out"));
    const std::string key_softmax_pd = key + "@softmax_pd";

    auto softmax_pd =
        std::static_pointer_cast<mkldnn::softmax_forward::primitive_desc>(
            dev_ctx.GetBlob(key_softmax_pd));
    PADDLE_ENFORCE(softmax_pd != nullptr,
                   "Fail to find softmax_pd in device context");

    // TODO(jczaja): Add layouts support when there is a need to do so
    // Two dimensional softmax does support NC format
    auto data_softmax_md = MKLDNNMemDesc(
        {softmax_tz}, platform::MKLDNNGetDataType<T>(), memory::format_tag::nc);
    auto diff_softmax_md = MKLDNNMemDesc(
        {softmax_tz}, platform::MKLDNNGetDataType<T>(), memory::format_tag::nc);
    // Normalization is made after innermost dimension eg. C out of NC
    auto softmax_bwd_desc =
        softmax_backward::desc(diff_softmax_md, data_softmax_md, 1 /* dim: C*/);
    auto softmax_bwd_pd =
        std::make_shared<mkldnn::softmax_backward::primitive_desc>(
            softmax_bwd_desc, mkldnn_engine, *softmax_pd);

    SoftmaxMKLDNNHandler handler(softmax_pd, softmax_bwd_pd, dev_ctx,
                                 mkldnn_engine, key);
    auto dst_memory_p =
        handler.AcquireDstMemory(data_softmax_md, to_void_cast<T>(dst_data));
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(
        diff_softmax_md, to_void_cast<T>(diff_dst_ptr));
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(
        diff_softmax_md, to_void_cast<T>(diff_src_ptr));

    // Get primitve from device context
    auto softmax_bwd_p = handler.AcquireSoftmaxBackward();

    mkldnn::stream astream(mkldnn_engine);
    softmax_bwd_p->execute(astream,
                           {{MKLDNN_ARG_DST, *dst_memory_p},
                            {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                            {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
    astream.wait();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>);
REGISTER_OP_KERNEL(softmax_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNGradKernel<float>);

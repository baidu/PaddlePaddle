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


#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/operators/requantize_op.h"
#include "paddle/fluid/framework/data_layout_transform.h"

namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

template <typename T>
class ReQuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    //auto* scale = ctx.Input<Tensor>("Scale");
    auto* output = ctx.Output<Tensor>("Output");
std::cout<<"this is requantize op!!!!!!!!!!"<<std::endl;
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();
 
    std::vector<primitive> pipeline;
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    mkldnn::memory::data_type src_dt = paddle::framework::ToMKLDNNDataType(input->type());
    mkldnn::memory::data_type dst_dt = mkldnn::memory::data_type::u8;//paddle::framework::ToMKLDNNDataType(output->type());
    mkldnn::memory::format src_fmt = memory::format::nhwc;//input->format();
    mkldnn::memory::format dst_fmt = memory::format::nhwc;//output->format();

    const T* input_data = input->data<T>();
    uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
    //T scale_data = *(scale->data<T>());
    std::vector<float> scale_data = {0.9999999}; //{*(scale->data<float>())};

    mkldnn::primitive_attr attri;
    int mask = 0;
    attri.set_output_scales(mask,scale_data);// scale_data);
    //attri.set_int_output_round_mode(round_nearest); //FIX ME

    auto src_md = platform::MKLDNNMemDesc(
            {src_tz}, src_dt, src_fmt); //FIX ME WITH S8
    auto src_pd = mkldnn::memory::primitive_desc(src_md, engine);
    auto src_memory = std::make_shared<mkldnn::memory>(src_pd, to_void_cast<T>(input_data));
    std::shared_ptr<primitive::at> src_memory_p = std::shared_ptr<primitive::at>(new primitive::at(*src_memory));

    auto dst_md = platform::MKLDNNMemDesc(
            {dst_tz}, dst_dt, dst_fmt);
    auto dst_pd = mkldnn::memory::primitive_desc(dst_md, engine);
    auto dst_memory = mkldnn::memory(dst_pd, to_void_cast<uint8_t>(output_data));
    
    auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
        new reorder::primitive_desc(src_pd, dst_pd, attri));   

for(int i=0; i<50; i++){
    printf("%d ", *(input_data+i));
}
printf("\n");fflush(stdout);
//for(int i=0; i<50; i++){
//    printf("%f ", *(input_data+i)/107.426);
//}
//printf("\n");fflush(stdout);
std::cout<<"scale = "<<scale_data[0]<<std::endl;
//for(int i=0; i<50; i++){
//    printf("%f ", *(output_data+i)/107.426);
//}
//printf("\n");fflush(stdout);

//    int is_sum = false;//ctx.Attr<int>("is_sum");
//    if(is_sum){
//std::cout<<"input fmt = "<<input->format()<<"  output fmt = "<<output->format()<<"output dt = "<<paddle::framework::ToMKLDNNDataType(output->type())<<std::endl;
//        output_data = (uint8_t*)input_data;
//std::cout<<"input fmt = "<<input->format()<<"  output fmt = "<<output->format()<<"output dt = "<<paddle::framework::ToMKLDNNDataType(output->type())<<std::endl;
//
//printf("after*************\n");
//for(int i=0; i<50; i++){
//    printf("%f ", *(output_data+i)/107.426);
//}
//printf("\n");fflush(stdout);
//
//    } else{
        auto reorder_p= std::shared_ptr<reorder>(new reorder(*reorder_pd, *src_memory_p, dst_memory));
        pipeline.push_back(*reorder_p);
        stream(stream::kind::eager).submit(pipeline).wait();
//    }
//uint8_t* output_data_2 = output->mutable_data<uint8_t>(ctx.GetPlace());
//for(int i=0; i<50; i++){
//    printf("%f ", *(output_data_2+i)/107.426);
//}
//printf("\n");fflush(stdout);
for(int i=0; i<50; i++){
    printf("%d ", *(output_data+i));
}
printf("\n");fflush(stdout);
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(dst_memory));
std::cout<<"input fmt = "<<input->format()<<"  output fmt = "<<output->format()<<"output dt = "<<paddle::framework::ToMKLDNNDataType(output->type())<<std::endl;
  }
};

framework::OpKernelType ReQuantOp::GetExpectedKernelType(const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_{framework::LibraryType::kPlain};
  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::Tensor>("Input")->type()),ctx.GetPlace(),layout_, library_);
}

void ReQuantOpMaker::Make() {
  AddInput("Input","input");
  AddInput("Scale","scale...");
  AddOutput("Output","output");
AddComment(R"DOC(
This op will requantize data from INT8 to INT8
)DOC");
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(requantize, ops::ReQuantOp, ops::ReQuantOpMaker, paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_KERNEL(requantize, MKLDNN, ::paddle::platform::CPUPlace, ops::ReQuantOpKernel<int8_t>);

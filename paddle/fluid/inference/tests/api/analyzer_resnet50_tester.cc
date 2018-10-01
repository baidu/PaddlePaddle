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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/api/timer.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(data_list, "", "Path to a file with a list of image files.");
DEFINE_string(data_dir, "", "Path to a directory with image files.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(iterations, 1, "How many times to repeat run.");
DEFINE_int32(skip_batch_num, 0, "How many minibatches to skip in statistics.");
// dimensions of imagenet images are assumed as default:
DEFINE_int32(height, 224, "Height of the image.");
DEFINE_int32(width, 224, "Width of the image.");
DEFINE_int32(channels, 3, "Width of the image.");
DEFINE_bool(use_fake_data, false, "Use fake data (1,2,...).");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(debug_display_images, false, "Show images in windows for debug.");
DECLARE_bool(profile);

namespace paddle {

struct DataReader {
  explicit DataReader(const std::string& data_list_path,
                      const std::string& data_dir_path, int resize_width,
                      int resize_height, int channels, bool convert_to_rgb)
      : data_list_path(data_list_path),
        data_dir_path(data_dir_path),
        file(std::ifstream(data_list_path)),
        width(resize_width),
        height(resize_height),
        channels(channels),
        convert_to_rgb(convert_to_rgb) {
    if (!file.is_open()) {
      throw std::invalid_argument("Cannot open data list file " +
                                  data_list_path);
    }

    if (data_dir_path.empty()) {
      throw std::invalid_argument(
          "Data directory must be set to use imagenet.");
    }

    if (channels != 3) {
      throw std::invalid_argument("Only 3 channel image loading supported");
    }
  }

  // return true if separator works or false otherwise
  bool SetSeparator(char separator) {
    sep = separator;

    std::string line;
    auto position = file.tellg();
    std::getline(file, line);
    file.clear();
    file.seekg(position);

    // test out
    std::vector<std::string> pieces;
    inference::split(line, separator, &pieces);

    return (pieces.size() == 2);
  }

  bool NextBatch(float* input, int64_t* label, int batch_size,
                 bool debug_display_images) {
    std::string line;

    for (int i = 0; i < batch_size; i++) {
      if (!std::getline(file, line)) return false;

      std::vector<std::string> pieces;
      inference::split(line, sep, &pieces);
      if (pieces.size() != 2) {
        std::stringstream ss;
        ss << "invalid number of separators '" << sep << "' found in line " << i
           << ":'" << line << "' of file " << data_list_path;
        throw std::runtime_error(ss.str());
      }

      auto filename = data_dir_path + pieces.at(0);
      label[i] = std::stoi(pieces.at(1));

      cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
      if (convert_to_rgb) {
        cv::cvtColor(image, image, CV_BGR2RGB);
      }

      if (image.data == nullptr) {
        std::string error_msg = "Couldn't open file " + filename;
        throw std::runtime_error(error_msg);
      }

      if (debug_display_images)
        cv::imshow(std::to_string(i) + " input image", image);

      cv::Mat image2;
      cv::resize(image, image2, cv::Size(width, height));

      cv::Mat fimage;
      image2.convertTo(fimage, CV_32FC3);

      fimage /= 255.f;

      cv::Scalar mean(0.406f, 0.456f, 0.485f);
      cv::Scalar std(0.225f, 0.224f, 0.229f);

      if (convert_to_rgb) {
        std::swap(mean[0], mean[2]);
        std::swap(std[0], std[2]);
      }

      std::vector<cv::Mat> fimage_channels;
      cv::split(fimage, fimage_channels);

      for (int c = 0; c < channels; c++) {
        fimage_channels[c] -= mean[c];
        fimage_channels[c] /= std[c];
        for (int row = 0; row < fimage.rows; ++row) {
          const float* fimage_begin = fimage_channels[c].ptr<const float>(row);
          const float* fimage_end = fimage_begin + fimage.cols;
          std::copy(fimage_begin, fimage_end,
                    input + row * fimage.cols + c * fimage.cols * fimage.rows +
                        i * 3 * fimage.cols * fimage.rows);
        }
      }
    }
    return true;
  }

  std::string data_list_path;
  std::string data_dir_path;
  std::ifstream file;
  int width;
  int height;
  int channels;
  bool convert_to_rgb;
  char sep{'\t'};
};

void drawImages(float* input, bool is_rgb) {
  if (FLAGS_debug_display_images) {
    for (int b = 0; b < FLAGS_batch_size; b++) {
      std::vector<cv::Mat> fimage_channels;
      for (int c = 0; c < FLAGS_channels; c++) {
        fimage_channels.emplace_back(
            cv::Size(FLAGS_width, FLAGS_height), CV_32FC1,
            input + FLAGS_width * FLAGS_height * c +
                FLAGS_width * FLAGS_height * FLAGS_channels * b);
      }
      cv::Mat mat;
      if (is_rgb) {
        std::swap(fimage_channels[0], fimage_channels[2]);
      }
      cv::merge(fimage_channels, mat);
      cv::imshow(std::to_string(b) + " output image", mat);
    }
    std::cout << "Press any key in image window or close it to continue"
              << std::endl;
    cv::waitKey(0);
  }
}

template <typename T>
void fill_data(T* data, unsigned int count) {
  for (unsigned int i = 0; i < count; ++i) {
    *(data + i) = i;
  }
}

template <>
void fill_data<float>(float* data, unsigned int count) {
  static unsigned int seed = std::random_device()();
  static std::minstd_rand engine(seed);
  float mean = 0;
  float std = 1;
  std::normal_distribution<float> dist(mean, std);
  for (unsigned int i = 0; i < count; ++i) {
    data[i] = dist(engine);
  }
}

template <typename T>
void SkipFirstNData(std::vector<T>& v, int n) {
  std::vector<T>(v.begin() + FLAGS_skip_batch_num, v.end()).swap(v);
}

template <typename T>
T FindAverage(const std::vector<T>& v) {
  CHECK_GE(v.size(), 0);
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T>
T FindPercentile(std::vector<T> v, int p) {
  CHECK_GE(v.size(), 0);
  std::sort(v.begin(), v.end());
  if (p == 100) return v.back();
  int i = v.size() * p / 100;
  return v[i];
}

template <typename T>
T FindStandardDev(std::vector<T> v) {
  CHECK_GE(v.size(), 0);
  T mean = FindAverage(v);
  T var = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    var += (v[i] - mean) * (v[i] - mean);
  }
  var /= v.size();
  T std = sqrt(var);
  return std;
}

void PostprocessBenchmarkData(std::vector<double> latencies,
                              std::vector<float> infer_accs,
                              std::vector<double> fpses, double total_time_sec,
                              int total_samples) {
  // get rid of the first FLAGS_skip_batch_num data
  SkipFirstNData(latencies, FLAGS_skip_batch_num);
  SkipFirstNData(infer_accs, FLAGS_skip_batch_num);
  SkipFirstNData(fpses, FLAGS_skip_batch_num);

  double lat_avg = FindAverage(latencies);
  float acc_avg = FindAverage(infer_accs);
  double fps_avg = FindAverage(fpses);

  double lat_pc99 = FindPercentile(latencies, 99);
  double fps_pc01 = FindPercentile(fpses, 1);

  double lat_std = FindStandardDev(latencies);
  double fps_std = FindStandardDev(fpses);

  float examples_per_sec = total_samples / total_time_sec;

  printf("\nAvg fps: %.5f, std fps: %.5f, fps for 99pc latency: %.5f\n",
         fps_avg, fps_std, fps_pc01);
  printf("Avg latency: %.5f ms, std latency: %.5f ms, 99pc latency: %.5f ms\n",
         lat_avg, lat_std, lat_pc99);
  printf("Total examples: %d, total time: %.5f sec, total examples/sec: %.5f\n",
         total_samples, total_time_sec, examples_per_sec);
  printf("Avg accuracy: %f\n\n", acc_avg);
}

void Main() {
  auto count = [](std::vector<int>& shapevec) {
    auto sum = shapevec.size() > 0 ? 1 : 0;
    for (unsigned int i = 0; i < shapevec.size(); ++i) {
      sum *= shapevec[i];
    }
    return sum;
  };

  // define input: input
  std::vector<int> shape;
  shape.push_back(FLAGS_batch_size);
  shape.push_back(FLAGS_channels);
  shape.push_back(FLAGS_height);
  shape.push_back(FLAGS_width);
  paddle::PaddleTensor input;
  input.name = "xx";
  input.shape = shape;

  // define input: label
  int label_size = FLAGS_batch_size;
  paddle::PaddleTensor input_label;
  input_label.data.Resize(label_size * sizeof(int64_t));
  input_label.name = "yy";
  input_label.shape = std::vector<int>({label_size, 1});
  input_label.dtype = paddle::PaddleDType::INT64;

  CHECK_GE(FLAGS_iterations, 0);
  CHECK_GE(FLAGS_skip_batch_num, 0);

  // reader instance for not fake data
  std::unique_ptr<DataReader> reader;
  bool convert_to_rgb = true;

  // Read first batch
  if (FLAGS_use_fake_data) {
    // create fake data
    input.data.Resize(count(shape) * sizeof(float));
    fill_data<float>(static_cast<float*>(input.data.data()), count(shape));

    input.dtype = paddle::PaddleDType::FLOAT32;

    std::cout << std::endl
              << "Executing model: " << FLAGS_infer_model << std::endl
              << "Batch Size: " << FLAGS_batch_size << std::endl
              << "Channels: " << FLAGS_channels << std::endl
              << "Height: " << FLAGS_height << std::endl
              << "Width: " << FLAGS_width << std::endl;

    // create fake label
    fill_data<int64_t>(static_cast<int64_t*>(input_label.data.data()),
                       label_size);
  } else {
    reader.reset(new DataReader(FLAGS_data_list, FLAGS_data_dir, FLAGS_width,
                                FLAGS_height, FLAGS_channels, convert_to_rgb));
    if (!reader->SetSeparator('\t')) reader->SetSeparator(' ');
    // get imagenet data and label
    input.data.Resize(count(shape) * sizeof(float));
    input.dtype = PaddleDType::FLOAT32;

    reader->NextBatch(static_cast<float*>(input.data.data()),
                      static_cast<int64_t*>(input_label.data.data()),
                      FLAGS_batch_size, FLAGS_debug_display_images);
  }

  // create predictor
  contrib::AnalysisConfig config;
  // MKLDNNAnalysisConfig config;
  config.model_dir = FLAGS_infer_model;
  // include mode: define which passes to include
  config.SetIncludeMode();
  config.use_gpu = false;
  config.enable_ir_optim = true;
  if (!FLAGS_skip_passes) {
    // add passes to execute keeping the order - without MKL-DNN
    config.ir_passes.push_back("conv_bn_fuse_pass");
    config.ir_passes.push_back("fc_fuse_pass");
#ifdef PADDLE_WITH_MKLDNN
    // add passes to execute with MKL-DNN
    config.ir_mkldnn_passes.push_back("conv_bn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_eltwiseadd_bn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_bias_mkldnn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_elementwise_add_mkldnn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_relu_mkldnn_fuse_pass");
    config.ir_mkldnn_passes.push_back("fc_fuse_pass");
#endif
  }
  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);

  // define output
  std::vector<PaddleTensor> output_slots;

  // run prediction
  inference::Timer timer;
  inference::Timer timer_total;
  std::vector<float> infer_accs;
  std::vector<double> batch_times;
  std::vector<double> fpses;
  for (int i = 0; i < FLAGS_iterations + FLAGS_skip_batch_num; i++) {
    if (i > 0) {
      if (FLAGS_use_fake_data) {
        fill_data<float>(static_cast<float*>(input.data.data()), count(shape));
        fill_data<int64_t>(static_cast<int64_t*>(input_label.data.data()),
                           label_size);
      } else {
        if (!reader->NextBatch(static_cast<float*>(input.data.data()),
                               static_cast<int64_t*>(input_label.data.data()),
                               FLAGS_batch_size, FLAGS_debug_display_images)) {
          std::cout << "No more full batches. stopping.";
          break;
        }
      }
    }

    drawImages(static_cast<float*>(input.data.data()), convert_to_rgb);

    if (i == FLAGS_skip_batch_num) {
      timer_total.tic();
      if (FLAGS_profile) {
        auto pf_state = paddle::platform::ProfilerState::kCPU;
        paddle::platform::EnableProfiler(pf_state);
      }
    }
    timer.tic();
    CHECK(predictor->Run({input, input_label}, &output_slots));
    double batch_time = timer.toc();
    CHECK_GE(output_slots.size(), 3UL);
    CHECK_EQ(output_slots[1].lod.size(), 0UL);
    CHECK_EQ(output_slots[1].dtype, paddle::PaddleDType::FLOAT32);
    batch_times.push_back(batch_time);
    float* acc1 = static_cast<float*>(output_slots[1].data.data());
    infer_accs.push_back(*acc1);
    double fps = FLAGS_batch_size * 1000 / batch_time;
    fpses.push_back(fps);
    std::string appx = (i < FLAGS_skip_batch_num) ? " (warm-up)" : "";
    std::cout << "Iteration: " << appx << i << ", "
              << "accuracy: " << *acc1 << ", "
              << "latency: " << batch_time << " ms, "
              << "fps: " << fps << std::endl;
  }

  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "/tmp/profiler");
  }

  double total_samples = FLAGS_iterations * FLAGS_batch_size;
  double total_time = timer_total.toc() / 1000;
  PostprocessBenchmarkData(batch_times, infer_accs, fpses, total_time,
                           total_samples);
}

TEST(resnet50, basic) { Main(); }

}  // namespace paddle

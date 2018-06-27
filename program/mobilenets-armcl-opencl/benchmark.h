/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#if defined(ARMCL_18_05_PLUS)
#include <arm_compute/graph.h>
#include <arm_compute/graph/nodes/Nodes.h>
#else
#include <arm_compute/graph/Graph.h>
#include <arm_compute/graph/Nodes.h>
#endif    
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CPP/CPPScheduler.h>
#include <arm_compute/runtime/Scheduler.h>
#include <support/ToolchainSupport.h>
#include <arm_compute/runtime/CL/CLTuner.h>
#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>

#include "GraphUtils.h"
#include "Utils.h"

#include <xopenme.h>

#include <cstdlib>
#include <iostream>
#include <string> 
#include <chrono>

enum GLOBAL_TIMER {
  X_TIMER_SETUP,
  X_TIMER_TEST,

  GLOBAL_TIMER_COUNT
};

enum GLOBAL_VAR {
  VAR_TIME_SETUP,
  VAR_TIME_TEST,
  VAR_TIME_IMG_LOAD_TOTAL,
  VAR_TIME_IMG_LOAD_AVG,
  VAR_TIME_CLASSIFY_TOTAL,
  VAR_TIME_CLASSIFY_AVG,

  GLOBAL_VAR_COUNT
};

using namespace std;
using namespace arm_compute;
using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

inline char path_separator()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

inline int getenv_i(const char* name, int def) {
    return getenv(name) ? atoi(getenv(name)) : def;
}

inline float getenv_f(const char* name, float def) {
    return getenv(name) ? atof(getenv(name)) : def;
}

inline void store_value_f(int index, const char* name, float value) {
    char* json_name = new char[strlen(name) + 6];
    sprintf(json_name, "\"%s\":%%f", name);
    xopenme_add_var_f(index, json_name, value);
    delete[] json_name;
}

inline int get_batch_size() {
    return getenv_i("CK_BATCH_SIZE", 1);
}

inline int get_batch_count() {
    return getenv_i("CK_BATCH_COUNT", 1);
}

inline const char* get_weights_path() {
    return getenv("CK_ENV_MOBILENET");
}

inline int get_image_size() {
    return getenv_i("CK_ENV_MOBILENET_RESOLUTION", 1);
}

inline const char* get_labels_file() {
  return getenv("CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT");
}

inline string get_mode_suffix() {
  ostringstream s;
  s << "-" << get_image_size() << "-" << get_batch_size() << "-" << get_batch_count() << ".txt";
  return s.str();
}

inline string get_images_list() {
  return getenv("CK_IMAGE_LIST") + get_mode_suffix();
}

inline string get_batches_list() {
  return getenv("CK_BATCH_LIST") + get_mode_suffix();
}

inline const char* get_result_dir() {
  return getenv("CK_RESULTS_DIR");
}

inline float get_multiplier() {
  return getenv_f("CK_ENV_MOBILENET_MULTIPLIER", 1);
}

#if defined(ARMCL_18_05_PLUS)
inline arm_compute::graph::ConvolutionMethod get_convolution_method() {
  auto method_name = getenv("CK_CONVOLUTION_METHOD");
  if (!method_name) {
      bool bifrost_target = (arm_compute::CLScheduler::get().target() == arm_compute::GPUTarget::BIFROST);
      return (bifrost_target ? arm_compute::graph::ConvolutionMethod::DIRECT : arm_compute::graph::ConvolutionMethod::GEMM);
  }
  // Try to get convolution method by its name
  if (strcmp(method_name, "DEFAULT") == 0) return arm_compute::graph::ConvolutionMethod::DEFAULT;
  if (strcmp(method_name, "GEMM") == 0) return arm_compute::graph::ConvolutionMethod::GEMM;
  if (strcmp(method_name, "DIRECT") == 0) return arm_compute::graph::ConvolutionMethod::DIRECT;
  if (strcmp(method_name, "WINOGRAD") == 0) return arm_compute::graph::ConvolutionMethod::WINOGRAD;
  // Try to get convolution method as integer value.
  // ConvolutionMethod enum has additional item comparing to ConvolutionMethodHint.
  // So we shift int value here to be consistent with old version: 0 = GEMM, 1 = DIRECT
  return static_cast<arm_compute::graph::ConvolutionMethod>(atoi(method_name)+1);
} 
#else
inline ConvolutionMethodHint get_convolution_method() {
  auto method_name = getenv("CK_CONVOLUTION_METHOD");
  if (!method_name) {
      bool bifrost_target = (arm_compute::CLScheduler::get().target() == arm_compute::GPUTarget::BIFROST);
      return (bifrost_target ? ConvolutionMethodHint::DIRECT : ConvolutionMethodHint::GEMM);
  }
  // Try to get convolution method by its name
  if (strcmp(method_name, "GEMM") == 0) return ConvolutionMethodHint::GEMM;
  if (strcmp(method_name, "DIRECT") == 0) return ConvolutionMethodHint::DIRECT;
  // Try to get convolution method as integer value.
  return static_cast<ConvolutionMethodHint>(atoi(method_name));
}
#endif

inline bool file_exists(const string& name) {
    ifstream f(name);
    return f.good();
}

#ifndef DATATYPE
#define DATATYPE DataType::F32
#endif

class CKPredictionSession {
public:
  const vector<string>& image_files() const { return _image_files; }
  const vector<string>& batch_files() const { return _batch_files; }
  int batch_index() const { return _batch_index; }
  size_t batch_size() const { return _batch_size; }
  size_t batch_count() const { return _batch_files.size(); }
  size_t image_size() const { return _image_size; }
  float total_load_images_time() const { return _total_load_images_time; }
  float total_prediction_time() const { return _total_prediction_time; }
  
  void init() {
    _batch_index = -1;
    _batch_size  = get_batch_size();
    _image_size  = get_image_size();
    _total_load_images_time = 0;
    _total_prediction_time = 0;

    load_file_list();
  }
  
  string get_next_batch_file() {
    if (_batch_index+1 >= _batch_files.size())
      return string();
    _batch_index++;
    return _batch_files[_batch_index];
  }
  
  void measure_begin() {
    _start_time = chrono::high_resolution_clock::now();
  }
  
  float measure_end() {
    auto finish_time = chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish_time - _start_time;
    return elapsed.count();
  }

  float measure_end_load_images() {
    auto duration = measure_end();
    _total_load_images_time += duration;
    return duration;
  }
  
  float measure_end_prediction() {
    auto duration = measure_end();
    _total_prediction_time += duration;
    return duration;
  }
  
private:
  int _batch_index;
  size_t _batch_size;
  size_t _image_size;
  vector<string> _image_files;
  vector<string> _batch_files;
  float _total_load_images_time;
  float _total_prediction_time;
  chrono::time_point<chrono::high_resolution_clock> _start_time;
  
  // TODO: Currently each batch consists of a single image, but it's not general case 
  // and additional work should be done to process real batches
  // https://github.com/ARM-software/ComputeLibrary/issues/355
  void load_file_list() {
    auto images_list = get_images_list();
    ifstream img_list(images_list);
    for (string file_name; !getline(img_list, file_name).fail();)
      _image_files.emplace_back(file_name);
    cout << "Image list file: " << images_list << endl;
    cout << "Image count in file: " << _image_files.size() << endl;

    auto batches_list = get_batches_list();
    ifstream batch_list(batches_list);
    for (string file_name; !getline(batch_list, file_name).fail();)
      _batch_files.emplace_back(file_name);
    cout << "Batch list file: " << batches_list << endl;
    cout << "Batch count in file: " << _batch_files.size() << endl;

    if (_batch_size != 1 || _image_files.size() != _batch_files.size())
       throw runtime_error("Only single image batches are currently supported");
  }
};

inline CKPredictionSession& session() {
  static CKPredictionSession s;
  return s;
}

#endif // BENCHMARK_H

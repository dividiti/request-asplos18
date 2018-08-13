/*
 * Copyright (c) 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNEuCTIOlN WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "tuner_convolution.h"

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <chrono>
#include <limits>
#include <string>

using namespace arm_compute;

CLTuner_Convolution::CLTuner_Convolution()
  : _lws_table() {
}

#if defined(ARMCL_18_05_PLUS)
void CLTuner_Convolution::tune_kernel_dynamic(ICLKernel &kernel) {
  ARM_COMPUTE_UNUSED(kernel);
}

void CLTuner_Convolution::tune_kernel_static(ICLKernel &kernel)
#else
void CLTuner_Convolution::tune_kernel(ICLKernel &kernel)
#endif
{
  // Get the configuration ID from the kernel
  const std::string &config_id = kernel.config_id();

  // Check if we need to find the Optimal LWS. If config_id is equal to default_config_id, the kernel does not require to be tuned
  if(config_id != arm_compute::default_config_id) {
    auto p = _lws_table.find(config_id);

    if(p == _lws_table.end()) {

      // Find the optimal LWS for the kernel
      cl::NDRange opt_lws = find_optimal_lws(kernel);

      // Insert the optimal LWS in the table
      _lws_table.emplace(config_id, opt_lws);

      // Set the LWS hint
      kernel.set_lws_hint(opt_lws);
    }
    else {
      // Set the LWS hint
      kernel.set_lws_hint(p->second);
    }
  }
}

inline cl::NDRange
benchmark_lws(cl::CommandQueue &queue, ICLKernel &kernel, double &opt_time,
              cl::NDRange opt_lws, cl::NDRange test_lws) {
  // Set the LWS hint to the given LWS.
  kernel.set_lws_hint(test_lws);
  // Start the clock.
  auto t_start = std::chrono::high_resolution_clock::now();
  // Run the kernel with the given LWS (if it is valid).
  kernel.run(kernel.window(), queue);
  CLScheduler::get().sync();
  // Stop the clock.
  auto t_stop = std::chrono::high_resolution_clock::now();
  // Check the elapsed execution time.
  std::chrono::duration<double, std::nano> fp_nano = t_stop - t_start;
  if(fp_nano.count() < opt_time) {
    opt_time = fp_nano.count();
    opt_lws  = test_lws;
  }
  // Return the optimal LWS.
  return opt_lws;
} // benchmark_lws()

cl::NDRange CLTuner_Convolution::find_optimal_lws(ICLKernel &kernel) {
  cl::CommandQueue queue = CLScheduler::get().queue();

  double opt_time = std::numeric_limits<double>::max();

  const char *ck_lws_tuner_kernel = getenv("CK_LWS_TUNER_KERNEL");
  const int kernel_id = ck_lws_tuner_kernel ? atoi(ck_lws_tuner_kernel) : 1; // tune "gemm"
  printf("\nkernel_id==%d (0 - im2col, 1 - gemm, 2 - col2im)\n", kernel_id);

  // Run the kernel once with the default configuration.
  const int bifrost_maximum_lws = 384;
  cl::NDRange opt_lws = cl::NDRange(bifrost_maximum_lws + 1, 1, 1);
  opt_lws = benchmark_lws(queue, kernel, opt_time, opt_lws, opt_lws);
  // Get the configuration ID from the kernel.
  const std::string &config_id = kernel.config_id();
  printf("config_id==%s\n", config_id.c_str());
  // Get the kernel name assuming it starts with "im2col".
  const std::string kernel_id_im2col = config_id.substr(0, std::strlen("im2col"));
  printf("kernel_id_im2col==%s\n", kernel_id_im2col.c_str());
  // Get the kernel name assuming it starts with "gemm".
  const std::string kernel_id_gemm = config_id.substr(0, std::strlen("gemm"));
  printf("kernel_id_gemm==%s\n", kernel_id_gemm.c_str());
  // Get the kernel name assuming it starts with "col2im".
  const std::string kernel_id_col2im = config_id.substr(0, std::strlen("col2im"));
  printf("kernel_id_col2im==%s\n", kernel_id_col2im.c_str());
  if(kernel_id_im2col == "im2col" && kernel_id == 0) {
    printf("Tuning %s (%s) ...\n", config_id.c_str(), kernel_id_im2col.c_str());
    for(int x = 1; x <= 4; x += 1)
      for(int y = 1; y <= 4; y += 1)
        for(int z = 1; z <= 4; z += 1)
          opt_lws = benchmark_lws(queue, kernel, opt_time, opt_lws, cl::NDRange(x, y, z));
  } // if(config_id == "im2col")
  else if(kernel_id_gemm == "gemm" && kernel_id == 1) {
    printf("Tuning %s (%s) ...\n", config_id.c_str(), kernel_id_gemm.c_str());
    for(int x = 1; x <= 4; x += 1)
      for(int y = 1; y <= 4; y += 1)
        for(int z = 1; z <= 1; z += 1)
          opt_lws = benchmark_lws(queue, kernel, opt_time, opt_lws, cl::NDRange(x, y, z));
  } // if(config_id == "gemm")
  else if(kernel_id_col2im == "col2im" && kernel_id == 2) {
    printf("Tuning %s (%s) ...\n", config_id.c_str(), kernel_id_col2im.c_str());
    for(int x = 1; x <= 4; x += 1)
      for(int y = 1; y <= 4; y += 1)
        for(int z = 1; z <= 4; z += 1)
          opt_lws = benchmark_lws(queue, kernel, opt_time, opt_lws, cl::NDRange(x, y, z));
  } // if(config_id == "col2im")

  return opt_lws;
}

void CLTuner_Convolution::import_lws_table(const std::unordered_map<std::string, cl::NDRange> &lws_table) {
  _lws_table.clear();
  _lws_table = lws_table;
}

const std::unordered_map<std::string, cl::NDRange> &CLTuner_Convolution::export_lws_table() {
  return _lws_table;
}

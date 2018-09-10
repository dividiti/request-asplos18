/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

// TODO: these headers should be moved to a common location (where?)
#include "../../../ck-tensorflow/program/image-classification-tflite/benchmark.h"
#include "../../../ck-math/program/armcl-classification-mobilenet/armcl_graph_common.h"

using namespace std;
using namespace CK;

void setup_mobilenet(GraphObject& graph,
                     unsigned int image_size,
                     float multiplier,
                     const std::string& weights_dir,
                     const float *input_data_buffer,
                     float *output_data_buffer,
                     CKDataLayout data_layout);


int main(int argc, const char **argv)
{
    try
    {
        init_benchmark();

        auto tuner_type = get_lws_tuner_type();
        auto tuner = get_lws_tuner(tuner_type);
        init_armcl(tuner.get());
        
        BenchmarkSettings settings;
        if (settings.batch_size != 1)
            throw runtime_error("Only single image batches are currently supported");

        int resolution = getenv_i("RUN_OPT_RESOLUTION");
        float multiplier = getenv_f("RUN_OPT_MULTIPLIER");
        auto data_layout = getenv_s("RUN_OPT_DATA_LAYOUT") == "NHWC" ? LAYOUT_NHWC : LAYOUT_NCHW;
        cout << "Data layout: " << (data_layout == LAYOUT_NCHW ? "NCHW" : "NHWC") << endl;

        vector<float> input(resolution * resolution * 3);
        vector<float> probes(1001);

        BenchmarkSession session(&settings);
        Benchmark<float, InNormalize, OutCopy> benchmark(&settings, input.data(), probes.data());
        benchmark.has_background_class = true;
        
        cout << "\nLoading graph..." << endl;
        GRAPH(graph, "MobileNetV1");
        measure_setup([&]
        {
            setup_mobilenet(graph, resolution, multiplier, settings.graph_file, input.data(), probes.data(), data_layout);

          // Do a warm up run for dynamic tuners (i.e. first tune here, then measure the best configuration later).
          if (tuner_type == CL_TUNER_DEFAULT) {
            graph.run(); // Input buffer allocated but filled with a trash, it doesn't matter for the first run
            arm_compute::CLScheduler::get().sync(); // Ensure that all OpenCL jobs have completed.
          }
        });

        cout << "\nProcessing batches..." << endl;
        measure_prediction([&]
        {
            while (session.get_next_batch())
            {
                session.measure_begin();
                benchmark.load_images(session.batch_files());
                session.measure_end_load_images();

                session.measure_begin();
                graph.run();
                session.measure_end_prediction();

                benchmark.save_results(session.batch_files());
            }
        });
        
        finish_benchmark(session);
    }
    catch (cl::Error &err)
    {
        cerr << "ERROR: " << err.what() << " (" << err.err() << ")" << endl;
        return -1;
    }
    catch (std::runtime_error &err)
    {
        cerr << "ERROR: " << err.what() << " " << (errno ? strerror(errno) : "") << endl;
        return -1;
    }
    catch (const string& error_message)
    {
        cerr << "ERROR: " << error_message << endl;
        return -1;
    }
    return EXIT_SUCCESS;
}

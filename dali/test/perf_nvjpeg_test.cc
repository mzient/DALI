#include "dali/test/dali_operator_test.h"

namespace dali {

TEST(NvJPEGPerf, Test) {
  std::vector<std::unique_ptr<Pipeline>> pipelines;
  int num_devices = 1;
  cudaGetDeviceCount(&num_devices);
  std::vector<int> good_devices;
  size_t max_mem = 0;
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop = {0};
    cudaGetDeviceProperties(&prop,  i);
    if (prop.totalGlobalMem > max_mem) {
      max_mem = prop.totalGlobalMem;
    }
  }
  size_t min_mem = max_mem / 2;
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop = {0};
    cudaGetDeviceProperties(&prop, i);
    if (prop.totalGlobalMem >= min_mem) {
      good_devices.push_back(i);
      std::cerr << "Using device " << i << ": " << prop.name << "\n";
    }
  }

  int old_device;
  cudaGetDevice(&old_device);
  for (int device_id : good_devices) {
    cudaSetDevice(device_id);
    pipelines.emplace_back(new Pipeline(256, 3, device_id));
    auto &pipe = *pipelines.back();
    OpSpec spec("nvJPEGDecoder");
    dali::testing::AddOperatorToPipeline(pipe, spec);
  }
  cudaSetDevice(old_device);
}

}  // namespace dali

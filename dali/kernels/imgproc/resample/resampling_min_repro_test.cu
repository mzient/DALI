#include "dali/kernels/imgproc/resample.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali::kernels {

void ShowResamplingResults() {
  using In = uint16_t;
  using Out = float;
  ResampleGPU<Out, In> res;
  TestTensorList<In> inp;
  TestTensorList<Out> out;
  int w = 5;
  int ow = 4;
  out.reshape(uniform_list_shape(1, { 1, ow, 1 }));
  inp.reshape(uniform_list_shape(1, { 1, w, 1 }));
  auto in_cpu = inp.cpu();
  for (int x = 0; x < w; x++) {
    in_cpu.data[0][x] = x == 1 ? 3 << 14 : 2 << 14;
  }
  DynamicScratchpad scratch;
  KernelContext ctx = {};
  ctx.gpu.stream = 0;
  ctx.scratchpad = &scratch;
  ResamplingParams2D params{};
  params[0].output_size = 1;
  params[0].mag_filter = params[0].min_filter = FilterDesc(ResamplingFilterType::Nearest, false, 1);

  params[1].output_size = ow;
  float r = DefaultFilterRadius(ResamplingFilterType::Cubic, true, w, ow);
  params[1].mag_filter = params[1].min_filter = FilterDesc(ResamplingFilterType::Cubic, true, r);
  (void)res.Setup(ctx, inp.gpu().to_static<3>(), make_span(&params, 1));
  res.Run(ctx, out.gpu().to_static<3>(), inp.gpu().to_static<3>(), make_span(&params, 1));
  auto out_cpu = out.cpu();
  CUDA_CALL(cudaDeviceSynchronize());
  for (int x = 0; x < ow; x++) {
    std::cout << (double)out_cpu[0].data[x] << " ";
  }
  std::cout << endl;

}

}

#if __STANDALONE

int main() {
  dali::kernels::ShowResamplingResults();
  return 0;
}

#else

#include <gtest/gtest.h>

namespace dali::kernels {

TEST(TestResamplingGPU, DumpResults) {
  dali::kernels::ShowResamplingResults();
}

}

#endif

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
  int h = 5;
  int oh = 4;
  out.reshape(uniform_list_shape(1, { oh, 1, 1 }));
  inp.reshape(uniform_list_shape(1, { h, 1, 1 }));
  auto in_cpu = inp.cpu();
  for (int y = 0; y < h; y++) {
    in_cpu.data[0][y] = y == 1 ? 52767 : 32767;
  }
  DynamicScratchpad scratch;
  KernelContext ctx = {};
  ctx.gpu.stream = 0;
  ctx.scratchpad = &scratch;
  ResamplingParams2D params{};
  params[0].output_size = oh;
  float r = DefaultFilterRadius(ResamplingFilterType::Cubic, true, h, oh);
  params[0].mag_filter = params[0].min_filter = FilterDesc(ResamplingFilterType::Cubic, true, r);

  params[1].output_size = 1;
  params[1].mag_filter = params[1].min_filter = FilterDesc(ResamplingFilterType::Nearest, false, 1);
  (void)res.Setup(ctx, inp.gpu().to_static<3>(), make_span(&params, 1));
  res.Run(ctx, out.gpu().to_static<3>(), inp.gpu().to_static<3>(), make_span(&params, 1));
  auto out_cpu = out.cpu();
  CUDA_CALL(cudaDeviceSynchronize());
  for (int y = 0; y < oh; y++) {
    printf("%f ", out_cpu[0].data[y]);
  }
  printf("\n");

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

// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RANDOM_RESIZED_CROP_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RANDOM_RESIZED_CROP_H_

#include <vector>
#include <random>
#include <memory>
#include <utility>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {

template <typename Backend>
class RandomResizedCrop : public Operator<Backend>
                        , protected ResizeBase<Backend>
                        , protected RandomCropAttr {
 public:
  explicit inline RandomResizedCrop(const OpSpec &spec)
      : Operator<Backend>(spec)
      , ResizeBase<Backend>(spec)
      , RandomCropAttr(spec)
      , interp_type_(spec.GetArgument<DALIInterpType>("interp_type"))
      , out_type_(spec.GetArgument<DALIDataType>("dtype")) {
    GetSingleOrRepeatedArg(spec, size_, "size", 2);
    InitParams(spec);
    BackendInit();
  }

  inline ~RandomResizedCrop() override = default;

  DISABLE_COPY_MOVE_ASSIGN(RandomResizedCrop);

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    auto &input = ws.template InputRef<Backend>(0);
    const auto &input_shape = input.shape();
    DALI_ENFORCE(input_shape.sample_dim() == 3, "Expects 3-dimensional HWC input.");

    DALIDataType out_type = out_type_;
    if (out_type == DALI_NO_TYPE)
      out_type = input.type().id();

    auto layout = this->InputLayout(ws, 0);

    int N = input_shape.num_samples();

    resample_params_.resize(N);

    int width_idx  = layout.find('W');
    int height_idx = layout.find('H');

    for (int sample_idx = 0; sample_idx < N; sample_idx++) {
      auto sample_shape = input_shape.tensor_shape_span(sample_idx);
      int H = sample_shape[height_idx];
      int W = sample_shape[width_idx];
      crops_[sample_idx] = GetCropWindowGenerator(sample_idx)({H, W}, "HW");
      resample_params_[sample_idx] = CalcResamplingParams(sample_idx);
    }

    output_desc.resize(1);
    this->SetupResize(output_desc[0].shape, out_type_, input_shape, input.type().id(),
                      make_cspan(resample_params_));
    output_desc[0].type = TypeTable::GetTypeInfo(out_type);
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override;

 private:
  void BackendInit();

  void CalcResamplingParams() {
    const int n = crops_.size();
    resample_params_.resize(n);
    for (int i = 0; i < n; i++)
      resample_params_[i] = CalcResamplingParams(i);
  }

  kernels::ResamplingParams2D CalcResamplingParams(int index) const {
    auto &wnd = crops_[index];
    auto params = shared_params_;
    params[0].roi = kernels::ResamplingParams::ROI(wnd.anchor[0], wnd.anchor[0]+wnd.shape[0]);
    params[1].roi = kernels::ResamplingParams::ROI(wnd.anchor[1], wnd.anchor[1]+wnd.shape[1]);
    return params;
  }

  void InitParams(const OpSpec &spec) {
    crops_.resize(batch_size_);
    shared_params_[0].output_size = size_[0];
    shared_params_[1].output_size = size_[1];
    shared_params_[0].min_filter = shared_params_[1].min_filter = this->min_filter_;
    shared_params_[0].mag_filter = shared_params_[1].mag_filter = this->mag_filter_;
  }

  std::vector<int> size_;
  DALIInterpType interp_type_;
  DALIDataType out_type_;
  kernels::ResamplingParams2D shared_params_;
  std::vector<kernels::ResamplingParams2D> resample_params_;
  std::vector<CropWindow> crops_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RANDOM_RESIZED_CROP_H_

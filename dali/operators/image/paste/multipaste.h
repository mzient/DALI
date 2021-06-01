// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_
#define DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/core/format.h"
#include "dali/util/crop_window.h"
#include "dali/pipeline/data/types.h"
#include "dali/kernels/imgproc/paste/paste_gpu_input.h"


namespace dali {

template <typename Backend, typename Actual>
class MultiPasteOp : public Operator<Backend> {
 public:
  DISABLE_COPY_MOVE_ASSIGN(MultiPasteOp);

  Actual &This() noexcept {
    return static_cast<Actual &>(*this);
  }

  const Actual &This() const noexcept {
    return static_cast<const Actual &>(*this);
  }

  template <typename T>
  using InList = TensorListView<detail::storage_tag_map_t<Backend>, const T>;

 protected:
  using Coords = TensorView<StorageCPU, const int, 1>;

  explicit MultiPasteOp(const OpSpec &spec)
      : Operator<Backend>(spec)
      , output_type_arg_(spec.GetArgument<DALIDataType>("dtype"))
      , output_type_(DALI_NO_TYPE)
      , input_type_(DALI_NO_TYPE)
      , output_size_("output_size", spec)
      , in_idx_("in_ids", spec)
      , in_anchors_("in_anchors", spec)
      , in_shapes_("shapes", spec)
      , out_anchors_("out_anchors", spec) {
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads_, max_batch_size_);
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  template<int ndim = 2>
  bool Intersects(const Coords& anchors1, const Coords& shapes1,
                  const Coords& anchors2, const Coords& shapes2) const {
    for (int i = 0; i < ndim; i++) {
      if (anchors1.data[i] + shapes1.data[i] <= anchors2.data[i]
          || anchors2.data[i] + shapes2.data[i] <= anchors1.data[i]) {
        return false;
      }
    }
    return true;
  }

  void AcquireArguments(const OpSpec &spec, const workspace_t<Backend> &ws) {
    const auto &images = ws.template InputRef<Backend>(0);

    auto curr_batch_size = ws.GetRequestedBatchSize(0);
    if (curr_batch_size == 0)
      return;

    output_size_.Acquire(spec, ws, curr_batch_size, true);
    in_idx_.Acquire(spec, ws, curr_batch_size, false);

    if (out_anchors_.IsDefined()) {
      out_anchors_.Acquire(spec, ws, curr_batch_size, false);
    }
    if (in_anchors_.IsDefined()) {
      in_anchors_.Acquire(spec, ws, curr_batch_size, false);
    }
    if (in_shapes_.IsDefined()) {
      in_shapes_.Acquire(spec, ws, curr_batch_size, false);
    }
    input_type_ = ws.template InputRef<Backend>(0).type().id();
    output_type_ =
        output_type_arg_ != DALI_NO_TYPE
        ? output_type_arg_
        : input_type_;

    raw_input_size_mem_.clear();
    const auto &input_shape = images.shape();

    raw_input_size_mem_.reserve(spatial_ndim_ * input_shape.num_samples());
    for (int i = 0; i < curr_batch_size; i++) {
      auto shape = input_shape.tensor_shape_span(i);
      for (int j = 0; j < spatial_ndim_; j++) {
        auto extent = static_cast<int>(shape[j]);
        raw_input_size_mem_.push_back(extent);
      }
    }

    no_intersections_.clear();
    for (int i = 0; i < curr_batch_size; i++) {
      const int64_t n_paste = in_idx_[i].shape[0];

      if (in_anchors_.IsDefined()) {
        DALI_ENFORCE(in_anchors_[i].shape[0] == n_paste,
                     "in_anchors must be same length as in_idx");
        DALI_ENFORCE(in_anchors_[i].shape[1] == spatial_ndim_,
                     make_string("Unexpected number of dimensions for ``in_anchors``. Expected ",
                     spatial_ndim_, ", got ", in_anchors_[i].shape[1]));
      }
      if (in_shapes_.IsDefined()) {
        DALI_ENFORCE(in_shapes_[i].shape[0] == n_paste, "in_shapes must be same length as in_idx");
        DALI_ENFORCE(in_shapes_[i].shape[1] == spatial_ndim_,
                     make_string("Unexpected number of dimensions for ``in_shapes``. Expected ",
                     spatial_ndim_, ", got ", in_shapes_[i].shape[1]));
      }
      if (out_anchors_.IsDefined()) {
        DALI_ENFORCE(out_anchors_[i].shape[0] == n_paste,
                     "out_anchors must be same length as in_idx");
        DALI_ENFORCE(out_anchors_[i].shape[1] == spatial_ndim_,
                     make_string("Unexpected number of dimensions for ``out_anchors``. Expected ",
                     spatial_ndim_, ", got ", out_anchors_[i].shape[1]));
      }

      bool found_intersection = false;
      for (int j = 0; j < n_paste; j++) {
        auto out_anchor = GetOutAnchors(i, j);
        auto in_anchor = GetInAnchors(i, j);
        auto j_idx = in_idx_[i].data[j];
        auto in_shape = Coords(raw_input_size_mem_.data() + spatial_ndim_ * j_idx,
                               dali::TensorShape<>(spatial_ndim_));
        const auto &shape = GetShape(i, j, in_shape);
        for (int k = 0; k < spatial_ndim_; k++) {
          DALI_ENFORCE(out_anchor.data[k] >= 0 && in_anchor.data[k] >= 0 &&
                       out_anchor.data[k] + shape.data[k] <= output_size_[i].data[k] &&
                       in_anchor.data[k] + shape.data[k] <= in_shape.data[k],
                       "Paste in/out coords should be within inout/output bounds.");
        }

        for (int k = 0; k < j; k++) {
          auto k_idx = in_idx_[i].data[k];
          if (Intersects(out_anchor, shape, GetOutAnchors(i, k), GetShape(
                  i, k, Coords(raw_input_size_mem_.data() + spatial_ndim_ * k_idx,
                               dali::TensorShape<>(spatial_ndim_))))) {
            found_intersection = true;
            break;
          }
        }
        if (found_intersection) {
          break;
        }
      }
      no_intersections_.push_back(!found_intersection);
    }
  }

  using Operator<Backend>::RunImpl;

  void RunImpl(workspace_t<Backend> &ws) override {
    const auto input_type_id = ws.template InputRef<Backend>(0).type().id();
    TYPE_SWITCH(input_type_id, type2id, InputType, (uint8_t, int16_t, int32_t, float), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
                This().template RunTyped<OutputType, InputType>(ws);
        ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
    ), DALI_FAIL(make_string("Unsupported input type: ", input_type_id)))  // NOLINT
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    const auto &images = ws.template InputRef<Backend>(0);
    int ndim = images.sample_dim();
    DALI_ENFORCE(ndim == 3, "MultiPaste supports only 2D data with channels (HWC)");

    int channel_dim = ndim - 1;
    int nsamples = images.ntensor();

    zeros_.resize(spatial_ndim_);
    zero_anchors_ = make_tensor_cpu<1>(zeros_.data(), { spatial_ndim_ });

    AcquireArguments(spec_, ws);

    output_desc.resize(1);
    output_desc[0].type = TypeTable::GetTypeInfo(output_type_);

    const auto &in_shape = images.shape();
    auto &out_shape = output_desc[0].shape;
    out_shape.resize(nsamples, in_shape.sample_dim());

    for (int i = 0; i < nsamples; i++) {
      TensorShape<> size(output_size_[i].data, output_size_[i].data + spatial_ndim_);
      auto out_sh = shape_cat(size, in_shape[i][channel_dim]);
      out_shape.set_tensor_shape(i, out_sh);
    }

    TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
            This().template SetupTyped<OutputType, InputType>(ws, out_shape);
        ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
    ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
    return true;
  }

  inline Coords GetInAnchors(int sample_num, int paste_num) const {
    return in_anchors_.IsDefined()
           ? subtensor(in_anchors_[sample_num], paste_num)
           : zero_anchors_;
  }

  inline Coords GetShape(int sample_num, int paste_num, Coords default_shape) const {
    return in_shapes_.IsDefined()
           ? subtensor(in_shapes_[sample_num], paste_num)
           : default_shape;
  }

  inline Coords GetOutAnchors(int sample_num, int paste_num) const {
    return out_anchors_.IsDefined()
           ? subtensor(out_anchors_[sample_num], paste_num)
           : zero_anchors_;
  }

  USE_OPERATOR_MEMBERS();
  DALIDataType output_type_arg_, output_type_, input_type_;

  ArgValue<int, 1> output_size_;

  ArgValue<int, 1> in_idx_;
  ArgValue<int, 2> in_anchors_;
  ArgValue<int, 2> in_shapes_;
  ArgValue<int, 2> out_anchors_;

  SmallVector<int, 4> zeros_;
  Coords zero_anchors_;

  const int spatial_ndim_ = 2;

  kernels::KernelManager kernel_manager_;

  vector<bool> no_intersections_;
  vector<int> raw_input_size_mem_;
};


class MultiPasteCPU : public MultiPasteOp<CPUBackend, MultiPasteCPU> {
 public:
  explicit MultiPasteCPU(const OpSpec &spec) : MultiPasteOp(spec) {}

 private:
  template<typename OutputType, typename InputType>
  void RunTyped(workspace_t<CPUBackend> &ws);

  template<typename OutputType, typename InputType>
  void SetupTyped(const workspace_t<CPUBackend> &ws,
                  const TensorListShape<> &out_shape);

  friend class MultiPasteOp<CPUBackend, MultiPasteCPU>;
};

class MultiPasteGPU : public MultiPasteOp<GPUBackend, MultiPasteGPU> {
 public:
  explicit MultiPasteGPU(const OpSpec &spec) : MultiPasteOp(spec) {}

 private:
  template<typename OutputType, typename InputType>
  void RunTyped(workspace_t<GPUBackend> &ws);

  template<typename OutputType, typename InputType>
  void SetupTyped(const workspace_t<GPUBackend> &ws,
                  const TensorListShape<> &out_shape);

  void InitSamples(const TensorListShape<> &out_shape);


  vector<kernels::paste::MultiPasteSampleInput<2>> samples_;

  friend class MultiPasteOp<GPUBackend, MultiPasteGPU>;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

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

#include <algorithm>
#include <cassert>
#include <random>
#include <utility>
#include "dali/core/static_switch.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/imgproc/structure/label_bbox.h"
#include "dali/operators/segmentation/random_object_bbox.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/imgproc/structure/connected_components.h"
#include "dali/kernels/imgproc/structure/label_bbox.h"

namespace dali {

using dali::kernels::OutTensorCPU;
using dali::kernels::InTensorCPU;
using dali::kernels::OutListCPU;
using dali::kernels::InListCPU;

using kernels::connected_components::LabelConnectedRegions;
using kernels::label_bbox::GetLabelBoundingBoxes;

DALI_SCHEMA(segmentation__RandomObjectBBox)
  .DocStr(R"(Randomly selects an object from a mask and returns its bounding box.

This operator takes a labeled segmentation map as its input. With probability ``foreground_prob``
it randomly selects a label (uniformly or according to the distribution given as ``weights``),
extracts connected blobs of pixels with the selected label and randomly selects one of them
(with additional constraints given as ``k_largest`` and ``threshold``).
With probability 1-foreground_prob, entire area of the input is returned.)")
  .NumInput(1)
  .OutputFn([](const OpSpec& spec) {
    return spec.GetArgument<string>("format") == "box" ? 1 : 2;
  })
  .AddOptionalArg("ignore_class", R"(If True, all objects are picked with equal probability,
regardless of the class they belong to. Otherwise, a class is picked first and then object is
randomly selected from this class.

This argument is incompatible with ``classes`` or ``class_weights``.

.. note::
  This flag only affects the probability with which blobs are selected. It does not cause
  blobs of different classes to be merged.)", false)
  .AddOptionalArg("foreground_prob", "Probability of selecting a foreground bounding box.", 1.0f,
    true)
  .AddOptionalArg<vector<int>>("classes", R"(List of labels considered as foreground.

If left unspecified, all labels not equal to ``background`` are considered foreground)",
    nullptr, true)
  .AddOptionalArg("background", R"(Background label.

If left unspecified, it's either 0 or any value not in ``classes``.)", 0, true)
  .AddOptionalArg<vector<float>>("class_weights", R"(Relative probabilities of foreground classes.

Each value corresponds to a class label in ``classes`` or a 1-based number if ``classes`` are
not specified.
The values are normalized so that they sum to 1.)", nullptr, true)
  .AddOptionalArg<int>("k_largest", "If specified, at most k largest bounding boxes are consider",
    nullptr)
  .AddOptionalArg<vector<int>>("threshold", R"(Minimum extent(s) of the bounding boxes to return.

If current class doesn't contain any bounding box that meets this condition, the largest one
is returned.)", nullptr)
  .AddOptionalArg("format", R"(Format in which the data is returned.

Possible choices are::
  * "anchor_shape" (the default) - there are two outputs: anchor and shape
  * "start_end" - there are two outputs - bounding box start and one-past-end coordinates
  * "box" - ther'es one output that contains concatenated start and end coordinates
)", "anchor_shape");


bool RandomObjectBBox::SetupImpl(vector<OutputDesc> &out_descs, const HostWorkspace &ws) {
  out_descs.resize(format_ == Out_Box ? 1 : 2);
  auto in_shape = ws.InputRef<CPUBackend>(0).shape();
  int ndim = in_shape.sample_dim();
  int N = in_shape.num_samples();
  AcquireArgs(ws, N, ndim);
  out_descs[0].type = TypeTable::GetTypeInfo(DALI_INT32);
  out_descs[0].shape = uniform_list_shape<DynamicDimensions>(
      N, TensorShape<1>{ format_ == Out_Box ? 2*ndim : ndim });
  for (size_t i = 1; i < out_descs.size(); i++)
    out_descs[i] = out_descs[0];
  return true;
}

void RandomObjectBBox::AcquireArgs(const HostWorkspace &ws, int N, int ndim) {
  background_.Acquire(spec_, ws, N);
  if (classes_.IsDefined())
    classes_.Acquire(spec_, ws, N);
  foreground_prob_.Acquire(spec_, ws, N);
  if (weights_.IsDefined())
    weights_.Acquire(spec_, ws, N);
  if (threshold_.IsDefined())
    threshold_.Acquire(spec_, ws, N, TensorShape<1>{ndim});

  if (weights_.IsDefined() && classes_.IsDefined()) {
    DALI_ENFORCE(weights_.get().shape == classes_.get().shape, make_string(
      "If both ``classes`` and ``weights`` are provided, their shapes must match. Got:"
      "\n  classes.shape  = ", classes_.get().shape,
      "\n  weights.shape  = ", weights_.get().shape));
  }
}


#define INPUT_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t)

template <typename T>
void RandomObjectBBox::FindLabels(std::unordered_set<int> &labels, const T *data, int64_t N) {
  if (!N)
    return;
  T prev = data[0];
  labels.insert(prev);
  for (int64_t i = 1; i < N; i++) {
    if (data[i] == prev)
      continue;  // skip runs of equal labels
    labels.insert(data[i]);
    prev = data[i];
  }
}

template <typename Out, typename In>
void FilterByLabel(Out *out, const In *in, int64_t N, In label) {
  for (int64_t i = 0; i < N; i++) {
    out[i] = in[i] == label;
  }
}

template <typename Out, typename In>
void FilterByLabel(const OutTensorCPU<Out> &out, const InTensorCPU<In> &in, same_as_t<In> label) {
  assert(out.shape == in.shape);
  int64_t N = in.num_elements();
  FilterByLabel(out.data, in.data, N, label);
}


template <typename Lo, typename Hi>
void StoreBox(const OutListCPU<int, 1> &out1,
              const OutListCPU<int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Lo &&start, Hi &&end) {
  assert(size(start) == size(end));
  int ndim = size(start);
  switch (format) {
    case RandomObjectBBox::Out_Box:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out1.data[sample_idx][i + ndim] = end[i];
      }
      break;
    case RandomObjectBBox::Out_AnchorShape:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out2.data[sample_idx][i] = end[i] - start[i];
      }
      break;
    case RandomObjectBBox::Out_StartEnd:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out2.data[sample_idx][i] = end[i];
      }
      break;
    default:
      assert(!"Unreachable code");
  }
}

template <typename Box>
void StoreBox(const OutListCPU<int, 1> &out1,
              const OutListCPU<int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Box &&box) {
  StoreBox(out1, out2, format, sample_idx, box.lo, box.hi);
}

void RandomObjectBBox::GetClassesAndWeightsArgs(
      ClassVec &classes, WeightVec &weights, int &background, int sample_idx) {
  background = background_[sample_idx].data[0];
  if (ignore_class_)
    return;  // we don't care about classes at all

  if (classes_.IsDefined()) {
    const auto &cls_tv = classes_[sample_idx];
    int ncls = cls_tv.shape[0];
    classes.resize(ncls);
    if (!weights_.IsDefined()) {
      weights.resize(ncls, 1.0f);
    }
    for (int i = 0; i < ncls; i++)
      classes[i] = cls_tv.data[i];
  }
  if (weights_.IsDefined()) {
    const auto &cls_tv = weights_[sample_idx];
    int ncls = cls_tv.shape[0];
    weights.resize(ncls);
    for (int i = 0; i < ncls; i++)
      weights[i] = cls_tv.data[i];
    if (!classes_.IsDefined()) {
      classes.resize(ncls);
      int cls = 0;
      for (int i = 0; i < ncls; i++, cls++) {
        if (cls == background)
          cls++;
        classes.push_back(cls);
      }
    }
  }
}

template <int ndim>
int RandomObjectBBox::PickBox(span<Box<ndim, int>> boxes, int sample_idx) {
  auto beg = boxes.begin();
  auto end = boxes.end();
  if (threshold_.IsDefined()) {
    vec<ndim, int> threshold;
    const int *thresh = threshold_[sample_idx].data;
    for (int i = 0; i < ndim; i++)
      threshold[i] = thresh[i];
    end = std::remove_if(beg, end, [threshold](auto box) {
      return any_coord(box.extent() < threshold);
    });
  }
  int n = beg - end;
  if (n == 0)
    return -1;

  if (k_largest_ >= 0) {
    SmallVector<std::pair<int64_t, int>, 32> vol_idx;
    vol_idx.resize(n);
    for (int i = 0; i < n; i++) {
      vol_idx[i] = { volume(boxes[i]), i };
    }
    std::sort(vol_idx.begin(), vol_idx.end());
    std::uniform_int_distribution<int> dist(0, std::min(n, k_largest_));
    return vol_idx[dist(rngs_[sample_idx])].second;
  } else {
    std::uniform_int_distribution<int> dist(0, n);
    return dist(rngs_[sample_idx]);
  }
}

bool RandomObjectBBox::PickBlob(SampleContext &ctx, int nblobs) {
  if (!nblobs)
    return false;

  int ndim = ctx.blobs.dim();
  ctx.box_data.resize(2*ndim*nblobs);

  VALUE_SWITCH(ndim, static_ndim, (1, 2, 3, 4, 5, 6),
    (
      auto *box_data = reinterpret_cast<Box<static_ndim, int>*>(ctx.box_data.data());
      auto boxes = make_span(box_data, nblobs);
      GetLabelBoundingBoxes(boxes, ctx.blobs.to_static<static_ndim>(), -1);
      int box_idx = PickBox(boxes, ctx.sample_idx);
      if (box_idx >= 0) {
        ctx.SelectBox(box_idx);
        return true;
      } else {
        return false;
      }
    ), (  // NOLINT
      DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim, "; must be 1..6"));
    )  // NOLINT
  );  // NOLINT
}

template <typename T>
bool RandomObjectBBox::PickForegroundBox(
      SampleContext &context, const InTensorCPU<T> &input) {
  GetClassesAndWeightsArgs(context.classes, context.weights, context.background,
                            context.sample_idx);
  if (ignore_class_) {
    int nblobs = LabelConnectedRegions(context.blobs, input, -1, context.background);
    return PickBlob(context, nblobs);
  } else {
    FindLabels(context.labels, input);

    context.labels.erase(context.background);

    if (!classes_.IsDefined() && !weights_.IsDefined()) {
      context.classes.clear();
      context.weights.clear();
      for (auto cls : context.labels) {
        context.classes.push_back(cls);
        context.weights.push_back(1);
      }
    } else {
      for (int i = 0; i < static_cast<int>(context.classes.size()); i++) {
        if (!context.labels.count(context.classes[i]))
          context.weights[i] = 0;  // label not present - reduce its weight to 0
      }
    }

    while (context.CalculateCDF()) {
      int class_idx = context.PickClassLabel(rngs_[context.sample_idx]);
      if (class_idx < 0)
        return false;
      int label = context.classes[class_idx];
      assert(label != context.background);
      FilterByLabel(context.filtered, input, label);

      int nblobs = LabelConnectedRegions<int64_t, uint8_t, -1>(
          context.blobs, context.filtered, -1, 0);

      if (!PickBlob(context, nblobs)) {
        // we couldn't find a satisfactory blob in this class, so let's exclude it and try again
        context.weights[class_idx] = 0;
      }
    }
    // we've run out of classes and still there's no good blob
    return false;
  }
}

bool RandomObjectBBox::PickForegroundBox(SampleContext &context) {
  bool ret = false;
  TYPE_SWITCH(context.input->type().id(), type2id, input_type, INPUT_TYPES,
    (ret = PickForegroundBox(context, view<const input_type>(*context.input));),
    (DALI_FAIL(make_string("Unsupported input type: ", context.input->type().id())))
  );  // NOLINT
  return ret;
}

void RandomObjectBBox::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  const auto &in_shape = input.shape();
  int N = in_shape.num_samples();
  int ndim = in_shape.sample_dim();
  auto &tp = ws.GetThreadPool();

  OutListCPU<int, 1> out1 = view<int, 1>(ws.OutputRef<CPUBackend>(0));
  OutListCPU<int, 1> out2;
  if (ws.NumOutput() > 1)
    out2 = view<int, 1>(ws.OutputRef<CPUBackend>(1));

  TensorShape<> default_anchor;
  default_anchor.resize(ndim);

  contexts_.resize(tp.size());

  std::uniform_real_distribution<> foreground(0, 1);
  for (int i = 0; i < N; i++) {
    bool fg = foreground(rngs_[i]) < foreground_prob_[i].data[0];
    if (!fg) {
      StoreBox(out1, out2, format_, i, default_anchor, in_shape[i]);
    } else {
      tp.AddWork([&, i](int thread_idx) {
        SampleContext &ctx = contexts_[thread_idx];
        ctx.Init(i, &input[i]);
        ctx.out1 = out1[i];
        if (out2.num_samples() > 0)
          ctx.out2 = out2[i];

        if (!PickForegroundBox(ctx))
          StoreBox(out1, out2, format_, i, default_anchor, in_shape[i]);
      }, volume(in_shape.tensor_shape_span(i)));
    }
  }
  tp.RunAll();
}


}  // namespace dali

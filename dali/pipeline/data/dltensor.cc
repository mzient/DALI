// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include "dali/pipeline/data/dltensor.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"

namespace dali {

DLDataType GetDLType(DALIDataType type) {
  DLDataType dl_type{};
  TYPE_SWITCH(type, type2id, T, (DALI_NUMERIC_TYPES_FP16, bool), (
    dl_type.bits = sizeof(T) * 8;
      dl_type.lanes = 1;
      if (dali::is_fp_or_half<T>::value) {
        dl_type.code = kDLFloat;
      } else if (std::is_same<T, bool>::value) {
        dl_type.code = kDLBool;
      } else if (std::is_unsigned<T>::value) {
        dl_type.code = kDLUInt;
      } else if (std::is_integral<T>::value) {
        dl_type.code = kDLInt;
      } else {
        DALI_FAIL(make_string("This data type (", type, ") cannot be handled by DLTensor."));
      }
  ), (DALI_FAIL(make_string("The element type ", type, " is not supported."))));  // NOLINT
  return dl_type;
}

void DLManagedTensorDeleter(DLManagedTensor *self) {
  delete static_cast<DLTensorResource*>(self->manager_ctx);
}

void DLMTensorPtrDeleter(DLManagedTensor* dlm_tensor_ptr) {
  if (dlm_tensor_ptr && dlm_tensor_ptr->deleter) {
    dlm_tensor_ptr->deleter(dlm_tensor_ptr);
  }
}

DLTensor MakeDLTensor(void *data, DALIDataType type,
                      std::optional<int> device_id,
                      const TensorShape<> &shape,
                      const TensorShape<> &strides) {
  DLTensor dl_tensor{};
  dl_tensor.data = data;
  dl_tensor.ndim = shape.size();
  dl_tensor.shape = shape.data();
  if (!strides.empty()) {
    dl_tensor.strides = strides.data();
  }
  if (device_id) {
    dl_tensor.device = {kDLCUDA, *device_id};
  } else {
    dl_tensor.device = {kDLCPU, 0};
  }
  dl_tensor.dtype = GetDLType(type);
  return dl_tensor;
}

DLMTensorPtr MakeDLTensor(void* data, DALIDataType type,
                          bool device, int device_id,
                          std::unique_ptr<DLTensorResource> resource) {
  DLManagedTensor *dlm_tensor_ptr = &resource->dlm_tensor;
  DLTensor &dl_tensor = dlm_tensor_ptr->dl_tensor;
  dl_tensor.data = data;
  dl_tensor.ndim = resource->shape.size();
  dl_tensor.shape = resource->shape.begin();
  if (!resource->strides.empty()) {
    dl_tensor.strides = resource->strides.data();
  }
  if (device) {
    dl_tensor.device = {kDLCUDA, device_id};
  } else {
    dl_tensor.device = {kDLCPU, 0};
  }
  dl_tensor.dtype = GetDLType(type);
  dlm_tensor_ptr->deleter = &DLManagedTensorDeleter;
  dlm_tensor_ptr->manager_ctx = resource.release();
  return {dlm_tensor_ptr, &DLMTensorPtrDeleter};
}

inline std::string to_string(const DLDataType &dl_type) {
  return std::string("{code: ")
    + (dl_type.code ? ((dl_type.code == 2) ? "kDLFloat" : "kDLUInt") : "kDLInt")
    + ", bits: " + std::to_string(dl_type.bits) + ", lanes: " + std::to_string(dl_type.lanes) + "}";
}

DALIDataType DLToDALIType(const DLDataType &dl_type) {
  DALI_ENFORCE(dl_type.lanes == 1,
               "DALI Tensors do not support types with the number of lanes other than 1");
  switch (dl_type.code) {
    case kDLUInt: {
      switch (dl_type.bits) {
        case 8: return DALI_UINT8;
        case 16: return DALI_UINT16;
        case 32: return DALI_UINT32;
        case 64: return DALI_UINT64;
      }
      break;
    }
    case kDLInt: {
      switch (dl_type.bits) {
        case 8: return DALI_INT8;
        case 16: return DALI_INT16;
        case 32: return DALI_INT32;
        case 64: return DALI_INT64;
      }
      break;
    }
    case kDLFloat: {
      switch (dl_type.bits) {
        case 16: return DALI_FLOAT16;
        case 32: return DALI_FLOAT;
        case 64: return DALI_FLOAT64;
      }
      break;
    }
    case kDLBool: {
      return DALI_BOOL;
      break;
    }
  }
  DALI_FAIL("Could not convert DLPack tensor of unsupported type " + to_string(dl_type));
}

}  // namespace dali

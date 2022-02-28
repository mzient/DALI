# Copyright (c) 2020, 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# it is enough to just import all functions from test_internals_operator_external_source
# nose will query for the methods available and will run them
# the test_internals_operator_external_source is 99% the same for cupy and numpy tests
# so it is better to store everything in one file and just call `use_cupy` to switch between the default numpy and cupy

from test_external_source_impl import *
use_torch(True)

# extra tests, GPU-specific
import torch
from test_utils import check_output
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline

def test_external_source_callback_torch_stream():
    with torch.cuda.stream(torch.cuda.Stream()):
        for attempt in range(10):
            t0 = torch.tensor([attempt * 100 + 1.5], dtype=torch.float32).cuda()
            increment = torch.tensor([10], dtype=torch.float32).cuda()
            pipe = Pipeline(1, 3, 0)

            def gen_batch():
                nonlocal t0
                t0 += increment
                return [t0]

            pipe.set_outputs(fn.external_source(gen_batch))
            pipe.build()

            for i in range(10):
                check_output(pipe.run(), [np.array([attempt * 100 + (i + 1) * 10 + 1.5], dtype=np.float32)])

def _test_cross_device(src, dst):
    import nvidia.dali.fn as fn
    pipe = Pipeline(1, 3, dst)
    data = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float32).cuda(device=dst)
    with pipe:
        pipe.set_outputs(fn.external_source([data], batch=False, device='gpu', cycle=True))

    pipe.build()
    for _ in range(10):
        pipe.run()

def test_cross_device():
    if torch.cuda.device_count() > 1:
        for src in [0,1]:
            for dst in [0,1]:
                yield _test_cross_device, src, dst

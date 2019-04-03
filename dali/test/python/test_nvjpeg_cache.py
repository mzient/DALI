# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from timeit import default_timer as timer
import numpy as np
import os
from numpy.testing import assert_array_equal, assert_allclose

seed = 1549361629

img_root = os.environ["DALI_EXTRA_PATH"]
image_dir = img_root + "/db/single/jpeg"
batch_size = 20

def compare(tl1, tl2):
    tl1_cpu = tl1.as_cpu()
    #tl2 = tl2.as_cpu()
    #print(tl1_cpu)

class nvJPEGDecoderPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, cache_size):
        super(nvJPEGDecoderPipeline, self).__init__(batch_size, num_threads, device_id, seed = seed)
        self.input = ops.FileReader(file_root = image_dir)
        policy = None
        if cache_size > 0:
          policy = "threshold"
        self.decode = ops.nvJPEGDecoder(device = 'mixed', output_type = types.RGB, cache_size = cache_size, cache_type = policy)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)

def test_nvjpeg_cached():
    ref_pipe = nvJPEGDecoderPipeline(batch_size, 1, 0, 0)
    ref_pipe.build()
    #cached_pipe = nvJPEGDecoderPipeline(batch_size, 1, 0, 100)
    #cached_pipe.build()

    ref_images, _ = ref_pipe.run()
    out_images, _ = ref_pipe.run()
    compare(ref_images, out_images)
    ref_images, _ = ref_pipe.run()
    out_images, _ = ref_pipe.run()
    compare(ref_images, out_images)
    ref_images, _ = ref_pipe.run()
    out_images, _ = ref_pipe.run()
    compare(ref_images, out_images)

def main():
    test_nvjpeg_cached()

if __name__ == '__main__':
    main()

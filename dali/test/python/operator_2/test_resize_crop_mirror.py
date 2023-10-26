# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from nose2.tools import params
from nvidia.dali import fn, types, pipeline_def
from test_utils import check_batch, get_dali_extra_path

test_data_root = get_dali_extra_path()
db_2d_folder = os.path.join(test_data_root, 'db', 'lmdb')


@pipeline_def(num_threads=4, batch_size=8, device_id=0, seed=1234)
def rcm_pipe(device, mode):
    files, labels = fn.readers.caffe(path=db_2d_folder, random_shuffle=True)
    images = fn.decoders.image(files, device="mixed" if device == "gpu" else "cpu")
    flip_x = fn.random.coin_flip(dtype=types.INT32)
    flip_y = fn.random.coin_flip(dtype=types.INT32)
    flip = flip_x | (flip_y * 2)
    if mode == "not_larger":  # avoid invalid crops
        size = fn.random.uniform(range=(480, 1000), shape=(2,), dtype=types.FLOAT)
    else:
        size = fn.random.uniform(range=(224, 480), shape=(2,), dtype=types.FLOAT)
    crop_w = fn.random.uniform(range=(100, 224), dtype=types.FLOAT)
    crop_h = fn.random.uniform(range=(100, 224), dtype=types.FLOAT)
    crop_x = fn.random.uniform(range=(0, 1))
    crop_y = fn.random.uniform(range=(0, 1))
    out = fn.resize_crop_mirror(images, size=size, mode=mode,
                                crop_w=crop_w, crop_h=crop_h, crop_pos_x=crop_x, crop_pos_y=crop_y,
                                mirror=flip)
    resized = fn.resize(images, size=size, mode=mode)
    cropped = fn.crop(resized, crop_w=crop_w, crop_h=crop_h, crop_pos_x=crop_x, crop_pos_y=crop_y)
    flipped = fn.flip(cropped, horizontal=flip_x, vertical=flip_y)
    return out, flipped


@params(("cpu", "not_larger"), ("cpu", None), ("gpu", "not_smaller"), ("gpu", "stretch"))
def test_vs_separate_ops(dev, mode):
    pipe = rcm_pipe(dev, mode)
    pipe.build()
    for _ in range(5):
        rcm, separate = pipe.run()
        check_batch(rcm, separate, len(rcm), 1e-3, 1)

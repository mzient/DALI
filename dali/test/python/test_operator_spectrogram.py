# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali as dali
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from functools import partial
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import ConstantDataIterator
from test_utils import get_dali_extra_path
import os
import librosa as librosa
import math

class SpectrogramPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, nfft, window_length, window_step,
                 window=None, num_threads=1, device_id=0):
        super(SpectrogramPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        window_fn = window(window_length).tolist() if window is not None else None
        self.fft = ops.Spectrogram(device = self.device,
                                   nfft = nfft,
                                   window_length = window_length,
                                   window_step = window_step,
                                   window_fn = window_fn,
                                   power = 2)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.fft(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        # randomly insert extra axis (channels?)
        r = np.random.randint(-1, 2)
        if r == 0:
            data = [x[np.newaxis,:] for x in data]
        elif r == 1:
            data = [x[:, np.newaxis] for x in data]

        self.feed_input(self.data, data)

def hann_win(n):
    hann = np.ones([n], dtype=np.float32)
    a = (2.0 * math.pi / n)
    for t in range(n):
        phase = a * (t + 0.5)
        hann[t] = 0.5 * (1.0 - math.cos(phase))
    return hann

def cos_win(n):
    phase = (np.arange(n) + 0.5) * (math.pi / n)
    return np.sin(phase).astype(np.float32)

def spectrogram_func_librosa(nfft, win_len, win_step, window, input_data):
    # Squeeze to 1d
    if len(input_data.shape) > 1:
        input_data = np.squeeze(input_data)

    if window is None:
        window = hann_win

    out = np.abs(librosa.stft(y=input_data, n_fft=nfft,
                              win_length=win_len, hop_length=win_step, window=window))**2

    # Alternative way to calculate the spectrogram:
    # out, _ = librosa.core.spectrum._spectrogram(
    #     y=input_data, n_fft=nfft, hop_length=win_step, window=hann_win, power=2)

    return out

class SpectrogramPythonPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, nfft, window_length, window_step, window=None,
                 num_threads=1, device_id=0, spectrogram_func=spectrogram_func_librosa):
        super(SpectrogramPythonPipeline, self).__init__(
              batch_size, num_threads, device_id,
              seed=12345, exec_async=False, exec_pipelined=False)
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(spectrogram_func, nfft, window_length, window_step, window)
        self.spectrogram = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.spectrogram(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def check_operator_spectrogram_vs_python(device, batch_size, input_shape,
                                         nfft, window_length, window_step):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        SpectrogramPipeline(device, batch_size, iter(eii1), nfft=nfft, window=None,
                            window_length=window_length, window_step=window_step),
        SpectrogramPythonPipeline(device, batch_size, iter(eii2), window=None, nfft=nfft,
                                  window_length=window_length, window_step=window_step),
        batch_size=batch_size, N_iterations=5, eps=1e-04)

def test_operator_spectrogram_vs_python():
    for device in ['cpu']:
        for batch_size in [3]:
            for nfft, window_length, window_step, shape in [(256, 256, 128, (1, 4096)),
                                                            (256, 256, 128, (4096,)),
                                                            (256, 256, 128, (4096, 1)),
                                                            (256, 256, 128, (1, 1, 4096, 1)),
                                                            (16, 16, 8, (1, 1000)),
                                                            (10, 10, 5, (1, 1000)),
                                                            ]:
                yield check_operator_spectrogram_vs_python, device, batch_size, shape, \
                    nfft, window_length, window_step

def check_operator_spectrogram_vs_python_wave_1d(device, batch_size, input_length,
                                                 nfft, window_length, window_step, window):
    f = 4000  # [Hz]
    sr = 44100  # [Hz]
    x = np.arange(input_length, dtype=np.float32)
    y = np.sin(2 * np.pi * f * x / sr)

    data1 = ConstantDataIterator(batch_size, y, dtype=np.float32)
    data2 = ConstantDataIterator(batch_size, y, dtype=np.float32)

    compare_pipelines(
        SpectrogramPipeline(device, batch_size, iter(data1), nfft=nfft,
                            window_length=window_length, window_step=window_step, window=window),
        SpectrogramPythonPipeline(device, batch_size, iter(data2),
                                  nfft=nfft, window_length=window_length, window_step=window_step,
                                  window=window),
        batch_size=batch_size, N_iterations=5, eps=1e-04)

def test_operator_spectrogram_vs_python_wave():
    for device in ['cpu', 'gpu']:
        for window in [None, hann_win, cos_win]:
            for batch_size in [3]:
                for nfft, window_length, window_step, length in [(256, 256, 128, 4096),
                                                                (128, 100, 61, 1000),
                                                                (10, 10, 5, 1000),
                                                                ]:
                    yield check_operator_spectrogram_vs_python_wave_1d, device, batch_size, \
                          length, nfft, window_length, window_step, window

for test in test_operator_spectrogram_vs_python_wave():
    test[0](*test[1:])

dali_extra = get_dali_extra_path()
audio_files = os.path.join(dali_extra, "db", "audio")


class AudioSpectrogramPipeline(Pipeline):
    def __init__(self, device, batch_size, nfft, window_length, window_step,
                 num_threads=1, device_id=0):
        super(AudioSpectrogramPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(device="cpu", file_root=audio_files)
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)
        self.fft = ops.Spectrogram(device=device,
                                   nfft=nfft,
                                   window_length=window_length,
                                   window_step=window_step,
                                   power=2)

    def define_graph(self):
        read, _ = self.input()
        audio, rate = self.decode(read)
        if self.fft.device == "gpu":
            audio = audio.gpu()
        spec = self.fft(audio)
        return spec


class AudioSpectrogramPythonPipeline(Pipeline):
    def __init__(self, batch_size, nfft, window_length, window_step,
                 num_threads=1, device_id=0, spectrogram_func=spectrogram_func_librosa):
        super(AudioSpectrogramPythonPipeline, self).__init__(
            batch_size, num_threads, device_id,
            seed=12345, exec_async=False, exec_pipelined=False)

        self.input = ops.FileReader(device="cpu", file_root=audio_files)
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)

        function = partial(spectrogram_func, nfft, window_length, window_step, None)
        self.spectrogram = ops.PythonFunction(function=function)

    def define_graph(self):
        read, _ = self.input()
        audio, rate = self.decode(read)
        out = self.spectrogram(audio)
        return out


def check_operator_decoder_and_spectrogram_vs_python(device, batch_size, nfft, window_length, window_step):
    compare_pipelines(
        AudioSpectrogramPipeline(device=device, batch_size=batch_size,
                                 nfft=nfft, window_length=window_length, window_step=window_step),
        AudioSpectrogramPythonPipeline(batch_size, nfft=nfft,
                                       window_length=window_length, window_step=window_step),
        batch_size=batch_size, N_iterations=5, eps=1e-04)


def test_operator_decoder_and_spectrogram():
    for device in ["cpu", "gpu"]:
        for batch_size in [3]:
            for nfft, window_length, window_step, shape in [(256, 256, 128, (1, 4096)),
                                                            (256, 256, 128, (4096,)),
                                                            (256, 256, 128, (4096, 1)),
                                                            (256, 256, 128, (1, 1, 4096, 1)),
                                                            (16, 16, 8, (1, 1000)),
                                                            (10, 10, 5, (1, 1000)),
                                                            ]:
                yield check_operator_decoder_and_spectrogram_vs_python, device, batch_size, \
                        nfft, window_length, window_step


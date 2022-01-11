# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from auto_scan_test import OPConvertAutoScanTest, BaseNet
from onnxbase import randtool
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.tensor.random.gaussian(
            inputs,
            mean=self.config["mean"],
            std=self.config["std"],
            dtype=self.config["out_dtype"])
        return x


class TestGaussianRandomConvert(OPConvertAutoScanTest):
    """
    api: paddle.tensor.random.gaussian
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=9), min_size=1, max_size=1))

        mean = draw(st.floats(min_value=-1.0, max_value=1.0))

        std = draw(st.floats(min_value=1.0, max_value=2.0))

        out_dtype = draw(st.sampled_from(["float32", "float64"]))

        def generator_data():
            input_data = randtool("int", 1, 10, input_shape)
            return input_data

        dtype = draw(st.sampled_from(["int32", "int64"]))

        config = {
            "op_names": ["gaussian_random"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": [15],
            "input_spec_shape": [],
            "mean": mean,
            "std": std,
            "out_dtype": out_dtype,
            "delta": 1e11,
            "rtol": 1e11
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net_list(BaseNet):
    """
    simple Net
    """

    def forward(self):
        """
        forward
        """
        x = paddle.tensor.random.gaussian(
            shape=self.config["shape"],
            mean=self.config["mean"],
            std=self.config["std"],
            dtype=self.config["out_dtype"])
        return x


class TestGaussianRandomConvert_list(OPConvertAutoScanTest):
    """
    api: paddle.tensor.random.gaussian
    OPset version: 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=5))

        mean = draw(st.floats(min_value=-1.0, max_value=1.0))

        std = draw(st.floats(min_value=1.0, max_value=2.0))

        out_dtype = draw(st.sampled_from(["float32", "float64"]))

        dtype = draw(st.sampled_from(["int32", "int64"]))

        config = {
            "op_names": ["gaussian_random"],
            "test_data_shapes": [],
            "test_data_types": [],
            "opset_version": [15],
            "input_spec_shape": [],
            "mean": mean,
            "std": std,
            "shape": input_shape,
            "out_dtype": out_dtype,
            "delta": 1e11,
            "rtol": 1e11
        }

        models = Net_list(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
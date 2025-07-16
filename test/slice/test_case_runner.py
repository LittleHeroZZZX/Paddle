# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import numpy as np
import torch

import paddle


def get_same_shape(shape1, shape2):
    size1 = len(shape1)
    size2 = len(shape2)
    new_shape = []
    i = 1
    while i <= min(size1, size2):
        if shape1[size1 - i] == shape2[size2 - i]:
            new_shape.append(shape1[size1 - i])
            i += 1
        else:
            break
    return new_shape[::-1]


class SliceTestCaseRunner:
    def __init__(self, test_case, place, place_id) -> None:
        self.place = place
        self.place_id = place_id

        self.input_shape = test_case.input_shape
        self.dtype = test_case.dtype
        self.framework = test_case.framework
        self.is_setitem = test_case.is_setitem
        self.index = test_case.index
        self.name = test_case.name
        if self.is_setitem:
            self.value_shape = (
                test_case.value_shape
                if test_case.value_shape is not None
                and len(test_case.value_shape) > 0
                else None
            )
            self.is_tensor = test_case.is_tensor

        self._do_prepare()

    def _do_prepare(self):
        self.np_input, self.tensor_input = self._prepare_data(
            self.input_shape,
            self.dtype,
            self.place,
            self.place_id,
            framework="paddle",
        )
        if self.is_setitem:
            np_infer_shape = self.np_input[self.index].shape
            if self.value_shape is None or len(self.value_shape) == 0:
                self.value_shape = np_infer_shape
            else:
                self.value_shape = get_same_shape(
                    np_infer_shape, self.value_shape
                )
            if self.is_tensor:
                self.np_value, self.value = self._prepare_data(
                    self.value_shape,
                    self.dtype,
                    self.place,
                    self.place_id,
                    framework="paddle",
                )
            else:
                self.value = 5

    def run(self):
        if self.is_setitem:
            self.tensor_input[self.index] = self.value
        else:
            self.output_tensor = self.tensor_input[self.index]

    def name(self):
        return self.name

    def _prepare_data(
        self,
        shape,
        dtype,
        place,
        place_id=None,
        framework="paddle",
        is_tensor=False,
    ) -> tuple[np.ndarray, paddle.Tensor | torch.Tensor]:

        if place == "cpu":
            torch_place = torch.device("cpu")
            paddle_place = paddle.CPUPlace()
        elif place == "gpu" or place == "cuda":
            place_id = 0 if place_id is None else place_id
            torch_place = torch.device(f"cuda:{place_id}")
            paddle_place = paddle.CUDAPlace(place_id)
        else:
            raise NotImplementedError(f"Unsupported place: {place}")

        np_data = np.random.randint(0, 100, size=shape)
        np_data.astype(dtype)
        if framework == "paddle":
            tensor = paddle.to_tensor(np_data.copy(), place=paddle_place)
        elif framework == "torch":
            tensor = torch.tensor(np_data.copy(), device=torch_place)
        else:
            raise NotImplementedError(f"Unsupported framework: {framework}")

        return np_data, tensor

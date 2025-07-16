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

#
import os
import sys

import numpy as np
import torch

import paddle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_case import SliceTestCase
from test_case_runner import SliceTestCaseRunner


class LayerEvalBM:
    """
    构建Layer评估的性能通用类
    """

    def __init__(self, test_case: SliceTestCase):
        """
        初始化
        """
        self.testcase = test_case
        self._set_env()
        self._set_seed()
        self._set_device()
        self._load_case()

    def _set_env(self):
        self.device = os.environ.get("SLICE_BENCHMARK_DEVICE", "gpu").lower()
        self.device_id = int(os.environ.get("SLICE_BENCHMARK_DEVICE_ID", "0"))
        self.n_repeat = int(
            os.environ.get("SLICE_BENCHMARK_REPEAT", "100")
        )  # 重复次数
        self.n_warmup = int(
            os.environ.get("SLICE_BENCHMARK_WARMUP", "3")
        )  # 预热次数
        self.seed = int(os.environ.get("PLT_BM_SEED", "2025"))  # 随机种子

    def _set_seed(self):
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        torch.manual_seed(self.seed)

    def _set_device(self):
        if self.device == "cpu":
            torch_device = "cpu"
            paddle_device = "cpu"
        elif self.device == "gpu" or self.device == "cuda":
            torch_device = f"cuda:{self.device_id}"
            paddle_device = f"gpu:{self.device_id}"
        else:
            raise NotImplementedError(f"Unsupported place: {self.device}")
        paddle.set_device(paddle_device)
        torch.device(torch_device)

    def _load_case(self):
        self.runner = SliceTestCaseRunner(
            self.testcase, self.device, self.device_id
        )

    def slice_perf(self):
        """slice perf"""

        start_event = [
            paddle.device.Event(enable_timing=True)
            for i in range(self.n_repeat)
        ]
        end_event = [
            paddle.device.Event(enable_timing=True)
            for i in range(self.n_repeat)
        ]
        paddle.device.synchronize()

        # warmup
        for _ in range(self.n_warmup):
            self.runner.run()

        paddle.device.synchronize()

        # 开始统计耗时
        for i in range(self.n_repeat):
            start_event[i].record()
            self.runner.run()
            end_event[i].record()
            paddle.device.synchronize()

        total_time_list = [
            s.elapsed_time(e) for s, e in zip(start_event, end_event)
        ]

        # print(self.runner.name+" :",total_time_list)
        ################################################################################################

        # if os.environ.get("PLT_BM_PLOT") == "True":
        #     save_pickle(data=total_time_list, filename="dy_eval_perf_" + self.layerfile)
        #     # 画图
        #     perf_by_step(
        #         data_list=total_time_list,
        #         step_scale=[0.1, 0.5, 1],
        #         filename="dy_eval_perf_" + self.layerfile + "_by_step",
        #     )

        ################################################################################################
        # # 事先写好的统计逻辑，例如掐头去尾求均值、取topk耗时，等等
        # time_res = eval(self.perf_statis)(data_list=total_time_list)
        # time_res = round(time_res * self.statis_times, self.statis_round)
        # ################################################################################################
        return total_time_list


if __name__ == "__main__":
    from test_case import generate_test_cases

    cases = generate_test_cases()
    # 这里可以添加测试代码
    for i in range(len(cases)):
        try:
            layer_eval = LayerEvalBM(cases[i])
            pd_res = layer_eval.slice_perf()
            pd_res = np.array(pd_res)
            i += 1
            layer_eval = LayerEvalBM(cases[i])
            torch_res = layer_eval.slice_perf()
            torch_res = np.array(torch_res)

            benchmark = np.around(pd_res / torch_res, 2)
            print(
                f"Case {i}-{layer_eval.runner.name}: Paddle vs Torch Benchmark: {benchmark}"
            )
        except Exception as e:
            print(f"Case {i}-{layer_eval.runner.name} failed to run.")
            print(e)

    # result = layer_eval.slice_perf()
    # print(result)

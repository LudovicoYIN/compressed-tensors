# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Container

import pytest
import torch
from compressed_tensors.offload.dispatch import dispatch_model
from tests.test_offload.conftest import DummyModel
from torch.fx import GraphModule, Tracer


class SeqeuentialTracer(Tracer):
    def __init__(
        self,
        sequential_targets: Container[torch.nn.Module],
        autowrap_modules: tuple = tuple(),
        autowrap_functions: tuple = tuple(),
        param_shapes_constant: bool = False,
    ):
        self.sequential_targets = sequential_targets
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def is_leaf_module(self, m, module_qualified_name):
        if m in self.sequential_targets:
            return True

        return super().is_leaf_module(m, module_qualified_name)

    def create_arg(self, a):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                # if a is p:
                # afaict we only need to do this for direct params
                if torch.equal(a, p.to(a.device)):
                    return self.create_node("get_attr", n, (), {})
            raise NameError("parameter is not a member of this module")
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        return super().create_arg(a)


@pytest.mark.integration
def test_trace_model():
    model = DummyModel()
    model = dispatch_model(model, "cuda:0")
    sample_input = torch.zeros(10, device="cuda:0")
    true_output = model(sample_input)

    graph = GraphModule(model, SeqeuentialTracer([model.fc1, model.fc2]).trace(model))
    assert graph is not None
    output = graph(sample_input)

    assert torch.equal(true_output, output)

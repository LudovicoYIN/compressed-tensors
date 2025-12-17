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

import gc
from weakref import ref

import pytest
import torch
from compressed_tensors.offload.cache.device import DeviceCache
from compressed_tensors.offload.module import OffloadMixin
from tests.testing_utils import requires_gpu


DEVICE = torch.device("cuda:0")


@pytest.fixture(scope="function")
def linear():
    return torch.nn.Linear(6, 7, bias=True)


@pytest.fixture(scope="function")
def cache():
    return DeviceCache(DEVICE)


@pytest.fixture(scope="function")
def input():
    return torch.zeros(6)


@pytest.mark.unit
@requires_gpu
def test_onloading(linear: torch.nn.Linear, cache: DeviceCache):
    weight = linear.weight
    bias = linear.bias

    linear = OffloadMixin.from_module(linear, cache)

    onloaded_weight = linear.weight
    onloaded_bias = linear.bias

    assert onloaded_weight.device == DEVICE
    assert onloaded_bias.device == DEVICE

    assert type(onloaded_weight) is type(weight)
    assert type(onloaded_bias) is type(bias)
    assert torch.equal(onloaded_weight.to(weight.device), weight)
    assert torch.equal(onloaded_bias.to(bias.device), bias)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect(linear: torch.nn.Linear, cache: DeviceCache):
    linear = OffloadMixin.from_module(linear, cache)
    weight_ref = ref(linear.weight)
    bias_ref = ref(linear.bias)

    del linear
    gc.collect()

    assert weight_ref() is None
    assert bias_ref() is None


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(linear: torch.nn.Linear, cache: DeviceCache):
    linear = OffloadMixin.from_module(linear, cache)

    outside_onloaded = linear.weight
    outside_onloaded_ref = ref(outside_onloaded)
    assert outside_onloaded.device == DEVICE

    with cache.disable_offloading():
        inside_onloaded = linear.weight
        inside_onloaded_ref = ref(inside_onloaded)
        assert inside_onloaded.device == DEVICE

        del outside_onloaded
        del inside_onloaded
        gc.collect()

        assert outside_onloaded_ref() is not None
        assert inside_onloaded_ref() is not None

    assert outside_onloaded_ref() is None
    assert inside_onloaded_ref() is None


@pytest.mark.unit
@requires_gpu
def test_disable_onloading(linear: torch.nn.Linear, cache: DeviceCache):
    weight = linear.weight
    linear = OffloadMixin.from_module(linear, cache)

    with cache.disable_onloading():
        onloaded = linear.weight
        assert onloaded is weight

    assert onloaded is weight


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("disable_offloading", [True, False])
def test_forward(
    linear: torch.nn.Linear, cache: DeviceCache, input: torch.Tensor, disable_offloading
):
    linear = OffloadMixin.from_module(linear, cache)

    output = linear.forward(input)
    output.device == DEVICE


# @pytest.mark.unit
# @requires_gpu
# def test_forward_disable_offload():
#     class MyModule(torch.nn.Module):
#         def __init__(self):
#             super.__init__(self)
#             self.a = torch.nn.Parameter(torch.zeros(6), requires_grad=False)

#         def forward(self, input: torch.Tensor):
#             pass

#     linear = OffloadMixin.from_module(linear, cache)


# pytest.mark.parametrize("with_offloading", [True, False])
# def test_call():


@pytest.mark.unit
@requires_gpu
def test_delete(linear: torch.nn.Linear, cache: DeviceCache):
    linear = OffloadMixin.from_module(linear, cache)
    weight_ref = ref(linear.weight)
    bias_ref = ref(linear.bias)

    del linear.weight
    del linear.bias
    gc.collect()

    assert weight_ref() is None
    assert bias_ref() is None

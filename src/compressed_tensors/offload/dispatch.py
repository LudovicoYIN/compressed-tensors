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
from typing import Optional, TypeVar

import torch

from .cache import DeviceCache
from .module import OffloadedModule


__all__ = ["dispatch_model", "remove_dispatch"]

ModelType = TypeVar("", bound=torch.nn.Module)


def dispatch_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: Optional[torch.device | str] = None,
    no_split_modules: Container[str] = tuple(),
) -> ModelType:
    if len(model._parameters) > 0:
        raise NotImplementedError(
            "Offloading is achieved by replacing modules which have direct parameters "
            "with new modules which have been wrapped. However, replacing the root "
            "can break functionality with previous implementation of `dispatch_model`. "
            "Please either remove any direct parameters to the model root, or refactor "
            "this function and its usages to use the new, wrapped root"
        )

    model = remove_dispatch(model)

    # each model shares a single shared cache because we have to
    # coordinate the onloading of shared tensors within the model
    cache = DeviceCache(onload_device, offload_device)
    memo = dict()
    for name, module in model.named_modules(remove_duplicate=False):
        # exclude wrapping the root
        if name == "" or isinstance(module, torch.nn.ModuleList):
            continue

        no_split = module.__class__.__name__ in no_split_modules
        offloaded_module = OffloadedModule.from_module(module, cache, no_split)

        model.set_submodule(name, offloaded_module)
        memo[module] = offloaded_module

    return model


def remove_dispatch(module: torch.nn.Module) -> torch.nn.Module:
    """
    Remove any existing dispatches from module

    :param module: module which may be dispatched with hf hooks
    :return: module without dispatch
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, OffloadedModule):
            if name == "":
                module = submodule._module
            else:
                module.set_submodule(name, submodule._module)

    return module

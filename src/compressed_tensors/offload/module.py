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

from contextlib import contextmanager
from typing import Any

import torch

from .cache.base import OffloadCache
from .utils import send_to_device


_offloaded_module_subclasses: dict[str, type] = dict()


class OffloadMixin(torch.nn.Module):
    _direct_attributes = {
        # core attributes
        "__class__",
        "__dict__",
        "__weakref__",
        # instance attributes
        "_module",
        "_cache",
        "_disable_offloading",
        "_offload_names",
        "disable_offloading",
        "disable_onloading",
        # these functions will return the wrapped `_module` unless we call with self
        "modules",
        "named_modules",
        "_modules",
        # call path
        "__call__",
        "_compiled_call_impl",
        "_call_impl",
        "forward",
    }

    def __init__(self, module: torch.nn.Module, cache: OffloadCache, no_split: bool):
        self._module = module
        self._cache = cache
        self._no_split = no_split
        self._offload_names = set(module.__dict__["_parameters"].keys())
        self._modules = module.__dict__["_modules"]

    def __getattribute__(self, name: str) -> object:
        if name in OffloadMixin._direct_attributes:
            return object.__getattribute__(self, name)

        elif name in self._offload_names:
            value = self._module._parameters[name]

            if value is not None:
                return self._cache[value]
            else:
                return None

        else:
            return getattr(self._module, name)

    def __setattr__(self, name: str, value: Any):
        if name in OffloadMixin._direct_attributes:
            return object.__setattr__(self, name, value)

        elif name in self._offload_names:
            old_value = self._module._parameters[name]

            if old_value is not None:
                self._cache[old_value] = value

        self._module.__setattr__(name, value)

    def __delattr__(self, name: str):
        if name in OffloadMixin._direct_attributes:
            return object.__delattr__(self, name)

        elif name in self._offload_names:
            old_value = self._module._parameters[name]

            if old_value is not None:
                del self._cache[old_value]

            self._offload_names.remove(name)

        self._module.__delattr__(name)

    def __call__(self, *args, **kwargs):
        args, kwargs = (
            send_to_device(args, self._cache.onload_device),
            send_to_device(kwargs, self._cache.onload_device),
        )

        if self._no_split:
            with self._cache.disable_offloading():
                return self._module.__call__.__func__(self, *args, **kwargs)
        else:
            return self._module.__call__.__func__(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        args, kwargs = (
            send_to_device(args, self._cache.onload_device),
            send_to_device(kwargs, self._cache.onload_device),
        )

        if self._no_split:
            with self._cache.disable_offloading():
                return self._module.forward.__func__(self, *args, **kwargs)
        else:
            return self._module.forward.__func__(self, *args, **kwargs)

    @contextmanager
    def disable_offloading(self):
        with self._cache.disable_offloading(self):
            yield

    @contextmanager
    def disable_onloading(self):
        with self._cache.disable_onloading(self):
            yield

    def execution_device(self) -> torch.device | str:
        return self._cache.onload_device

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        cache: OffloadCache,
        no_split: bool = False,
    ):
        class_name = module.__class__.__name__
        if class_name not in _offloaded_module_subclasses:
            _offloaded_module_subclasses[class_name] = make_offload_module_subclass(
                module.__class__
            )

        return _offloaded_module_subclasses[class_name](module, cache, no_split)


def make_offload_module_subclass(parent_cls: type) -> type:
    subclass = type(f"Offloaded{parent_cls.__name__}", (OffloadMixin, parent_cls), {})
    subclass.__name__ = parent_cls.__name__

    assert issubclass(subclass, parent_cls)
    return subclass

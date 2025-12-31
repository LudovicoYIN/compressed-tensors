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

import contextlib
from collections.abc import MutableMapping
from abc import ABC, abstractmethod
from typing import Literal, Optional
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from compressed_tensors.utils.global_access import GlobalAccess


class OffloadCache(GlobalAccess, MutableMapping):
    onload_device: torch.device | str
    offload_device: Optional[torch.device | str]

    @classmethod
    def from_devices(
        cls,
        onload_device: torch.device | str,
        offload_device: Optional[torch.device | str | Literal["disk"]] = None,
        distributed: Optional[bool] = None,
    ):
        from compressed_tensors.offload.cache.cpu import CPUCache

        if distributed is None:
            distributed = dist.is_available() and dist.is_initialized()

        if offload_device == torch.device("cpu") and not distributed:
            return CPUCache(onload_device)
        else:
            raise NotImplementedError(
                f"Offload of type {offload_device} and "
                f"distributed={distributed} has not been implemented"
            )

    def __init__(self, onload_device: torch.device | str):
        self.onload_device = onload_device
        self.offload_device = torch.device("cpu")

        # names -> offloaded tensors
        self.module = None
        self.offloaded_values: dict[tuple[torch.nn.Module, str], torch.Tensor] = dict()

        # offloaded tensors -> onloaded tensors
        self.onload_values: WeakValueDictionary[
            torch.Tensor, torch.Tensor
        ] = WeakValueDictionary()

        # strong ref to values to disable offloading
        self.keep_onloaded_values: set[torch.Tensor] = set()

    def curry_module(self, module: torch.nn.Module):
        copy = self.__class__(onload_device=self.onload_device)
        copy.onload_device = self.onload_device
        copy.offload_device = self.offload_device
        copy.offloaded_values = self.offloaded_values
        copy.onload_values = self.onload_values
        copy.keep_onloaded_values = self.keep_onloaded_values

        copy.module = module  # change prefix, shallow copy rest

        return copy

    @abstractmethod
    def onload(self, key: torch.Tensor) -> torch.Tensor:
        """
        Given an offloaded value, returns, onloaded version of that tensor

        :param key: offloaded tensor
        :return: onloaded tensor
        """
        # IMPL: return send_tensors(key, device=self.onload_device, copy=True)
        raise NotImplementedError()

    @abstractmethod
    def offload(self, value: torch.Tensor) -> torch.Tensor:
        """
        Given an onloaded value, returns the offloaded version of that tensor

        :param key: tensor to offload
        :return: offloaded tensor
        """
        # IMPL: return send_tensors(value, device=self.offload_device, copy=True)
        raise NotImplementedError()

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        :param key: offloaded tensor to be onloaded
        :return: onloaded tensor
        """
        offloaded = self.offloaded_values[self.module, key]
        if offloaded is None:
            return None

        # onload value, potentially from cache
        if offloaded not in self.onload_values:

            # onload value from (cpu)
            onloaded_value = self.onload(offloaded)
            self.onload_values[offloaded] = onloaded_value

        else:
            onloaded_value = self.onload_values[offloaded]

        return onloaded_value
    
    def __contains__(self, key) -> bool:
        return (self.module, key) in self.offloaded_values
    
    def __iter__(self):
        return iter([(module, key) for module, key in self.offloaded_values])

    def __len__(self):
        return len(self.offloaded_values)

    def __setitem__(self, key: str, value: torch.Tensor):
        """
        :param key: offloaded tensor whose value will be updated
        :param value: value used to update
        """
        # update data
        print("__setitem__")
        self.offloaded_values[self.module, key] = value

    def __delitem__(self, key: str):
        """
        :param key: offloaded tensor to be removed from the cache
        """
        del self.offloaded_values[self.module, key]


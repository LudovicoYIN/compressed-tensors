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

from abc import ABC, abstractmethod
from typing import Any

import torch
from compressed_tensors.utils.global_access import GlobalAccess


class OffloadCache(GlobalAccess, ABC):
    """
    Abstract base class for offload cache. Tensors are put into the cache via `offload`,
    and tensors are retrieved from the cache via `__getitem__`.
    """

    onload_device: torch.device | str

    @abstractmethod
    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """
        :param key: offloaded tensor to be onloaded
        :return: onloaded tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def __delitem__(self, key: torch.Tensor):
        """
        :param key: offloaded tensor to be removed from the cache
        """
        raise NotImplementedError()

    @abstractmethod
    def offload(self, key: torch.Tensor) -> torch.Tensor:
        """
        TODO
        FYI cache cannot be responsible for offloading
        unless maybe implement a new method

        DeviceCache:
        cpu_tensor = cache.offload(gpu_tensor)

        DiskCache:
        meta_tensor = cache.offload(gpu_tensor)
        """
        raise NotImplementedError()

    @abstractmethod
    def disable_offloading(self):
        raise NotImplementedError()

    @abstractmethod
    def disable_onloading(self):
        raise NotImplementedError()

    def __setitem__(self, key: torch.Tensor, value: Any):
        raise ValueError(
            "Cannot set item for OffloadCache. "
            "Please use `OffloadCache.offload` instead"
        )

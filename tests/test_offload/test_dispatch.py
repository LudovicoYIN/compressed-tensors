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

import pytest
from compressed_tensors.offload.dispatch import dispatch_model
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.mark.integration
@requires_gpu
def test_dispatch_llama_1b():

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    model = dispatch_model(model, "cuda:0", ["LlamaDecoderLayer"])

    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to("cuda:0") for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=15)
    print(tokenizer.batch_decode(output))

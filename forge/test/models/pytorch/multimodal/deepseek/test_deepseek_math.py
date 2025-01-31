# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        return self.model(input_tensor, max_new_tokens=100).logits


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepseek-math-7b-instruct"])
def test_deepseek_math_pytorch(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    model_name = f"deepseek-ai/{variant}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    framework_model = Wrapper(model)

    messages = [
        {
            "role": "user",
            "content": "what is the integral of x^2 from 0 to 2?\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=[input_tensor], module_name=module_name)

    # Model Verification
    verify([input_tensor], framework_model, compiled_model)

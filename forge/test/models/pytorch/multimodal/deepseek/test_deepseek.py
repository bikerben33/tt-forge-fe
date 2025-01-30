# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM

import forge
from forge.verify.verify import verify

from test.models.pytorch.multimodal.deepseek.utils.io import load_pil_images
from test.models.pytorch.multimodal.deepseek.utils.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)
from test.models.utils import Framework, Source, Task, build_module_name


def generate_model_deepseek_vl_pytorch(variant):
    model_path = variant
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.eos_token_id = tokenizer.eos_token_id
            self.bos_token_id = tokenizer.bos_token_id
            self.pad_token_id = tokenizer.pad_token_id

        def forward(self, inputs_embeds, attention_mask):
            return self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

    framework_model = Wrapper(vl_gpt)

    return framework_model, vl_gpt, vl_chat_processor, tokenizer


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepseek-ai/deepseek-vl-1.3b-base"])  # , "deepseek-ai/deepseek-vl-1.3b-chat"])
def test_deepseek_vl_pytorch(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Get the model, processor, and tokenizer
    framework_model, vl_gpt, vl_chat_processor, tokenizer = generate_model_deepseek_vl_pytorch(variant)

    # Single image conversation example
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Describe each stage of this image.",
            "images": [
                "/proj_sw/user_dev/mramanathan/lab72_jan29_forge/tt-forge-fe/forge/test/models/pytorch/multimodal/deepseek/image/training_pipelines.jpg"
            ],
        },
        {"role": "Assistant", "content": ""},
    ]

    # Load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(
        vl_gpt.device
    )

    # Run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    attention_mask = prepare_inputs.attention_mask
    inputs = [inputs_embeds, attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

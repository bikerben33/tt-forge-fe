import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify

@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
    ],
)
def test_in_place(shape):
    class Inplace(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + 1 
            x += 2

            return x + y
        
    input = torch.zeros(shape, requires_grad=False)
    input = torch.zeros(shape)
    torch_input = input.detach().clone()
    inputs = [input]

    framework_model = Inplace()
    y = framework_model(torch_input)
    
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="inplace")
    tty = compiled_model(*inputs)

    print(f"PyTorch output:\n {y}")
    print(f"TT output:\n {tty}")


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
    ],
)
def test_out_of_place(shape):
    class Inplace(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + 1 
            x = x + 2

            return x + y
        
    input = torch.zeros(shape, requires_grad=False)
    input = torch.zeros(shape)
    torch_input = input.detach().clone()
    inputs = [input]

    framework_model = Inplace()
    y = framework_model(torch_input)
    
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="inplace")
    tty = compiled_model(*inputs)

    print(f"PyTorch output:\n {y}")
    print(f"Tenstorrent output:\n {tty}")


@pytest.mark.parametrize(
    "input_tensor",
    [
        pytest.param(
            torch.ones(1, 18, dtype=torch.float32),
            id="simplified_case",
        ),
    ],
)
def test_minimal_bool_indexing(input_tensor): # decomposes multiple ops one of which is argwhere
    class MinimalBooleanIndexModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        def forward(self, x):
            # Create simple mask
            mask = x < self.threshold
            
            # Perform basic boolean indexing
            x[mask] = 7.0  # Simple assignment (modifies input tensor in-place)
            
            return x

    input_tensor[0, 1] = 0.0
    input_tensor[0, 2] = 0.0
    input_tensor[0, 3] = 0.0

    inputs = [input_tensor]
    framework_model = MinimalBooleanIndexModule()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)
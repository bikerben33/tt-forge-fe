import pytest
import torch
import tensorflow as tf
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.compare import compare_with_golden


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
    framework_input = input.detach().clone()
    inputs = [input]

    framework_model = Inplace()
    y = framework_model(framework_input)
    
    
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
def test_in_place_tf(shape):
    class Inplace(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            y = x + 1
            x += 2
            return x + y
        
    input = tf.zeros(shape)
    framework_input = tf.identity(input)
    inputs = [input]

    framework_model = Inplace()
    y = framework_model(framework_input)
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="inplace")
    tty = compiled_model(*inputs)

    print(f"TensorFlow output:\n {y}")
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


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
    ],
)
def test_in_place_backward(shape):
    class MatmulParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.rand(1024, 1024))
            nn.init.xavier_uniform_(self.p)

        def forward(self, x):
            y = torch.matmul(x, self.p)
            x += 2
            return y 

    model = MatmulParam()
    shape = (1, 1024)
    inputs = torch.rand(shape, requires_grad=True)
    # Fake targets
    target = torch.zeros(shape)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    tt_model = forge.compile(model, sample_inputs=[torch.rand(shape)], optimizer=optimizer)

    model.train()
    output = tt_model(inputs)
    # golden = model(inputs)

    output = [co.to("cpu") for co in output]
    # assert compare_with_golden(golden=golden, calculated=output[0])

    optimizer.zero_grad()

    loss = loss_fn(output[0], target)
    loss.backward()

    # golden_loss = loss_fn(golden, target)
    print(f"loss: {loss}")
    # print(f"golden_loss: {golden_loss}")
    print(f"output.grad: {output[0].grad}")

    loss_grad = output[0].grad
    assert loss_grad is not None
    grad = tt_model.backward()

    # HACK to run the optimizer step
    # i'm not sure what's the right way to tie the torch optimizer to our params,
    # but this can be done automatically after backward() (hidden from user)
    model.p.grad = grad[0]

    optimizer.step()
    
from src.mnist_linux_proj.model import MyAwesomeModel
import torch
import pytest

# def test_error_on_wrong_shape():
#     model = MyAwesomeModel()
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         model(torch.randn(1,1,3,3))
#     with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
#         model(torch.randn(1,28,28))

def test_model_forward():
    model = MyAwesomeModel()
    x = torch.randn(4, 1, 28, 28)  # batch of 4 images
    y_hat = model(x)
    assert y_hat.shape == (4, 10)  # batch of 4 outputs for 10 classes
    assert not torch.isnan(y_hat).any(), "Output contains NaNs"
def test_model_parameters():
    model = MyAwesomeModel()
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    for p in model.parameters():
        assert p.requires_grad, "All model parameters should require gradients"
def test_model_training_step():
    model = MyAwesomeModel()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    batch = (x, y)
    loss = model.training_step(batch)
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor), "Training step should return a tensor loss"
def test_model_validation_step():
    model = MyAwesomeModel()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    batch = (x, y)
    loss = model.validation_step(batch)
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor), "Validation step should return a tensor loss"
def test_model_test_step():
    model = MyAwesomeModel()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    batch = (x, y)
    loss = model.test_step(batch)
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor), "Test step should return a tensor loss"
def test_configure_optimizers():
    model = MyAwesomeModel()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    params = list(model.parameters())
    opt_params = list(optimizer.param_groups[0]['params'])
    assert params == opt_params
    assert optimizer.defaults['lr'] == 1e-3, "Default learning rate should be 1e-3"


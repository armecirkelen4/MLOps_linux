from src.mnist_linux_proj.train import train
import torch

def test_train_function():
    # check that train runs without errors and returns a model
    model = train(epochs=1, batch_size=16)
    assert model is not None
    # check that the model has been trained for at least one epoch
    assert model.current_epoch >= 1 
    # check that the model has learned something (loss should be a finite number)
    train_loss = model.logged_metrics.get("train_loss")
    assert train_loss is not None
    assert torch.isfinite(torch.tensor(train_loss))


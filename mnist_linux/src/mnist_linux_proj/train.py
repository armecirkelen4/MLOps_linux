import matplotlib.pyplot as plt
import torch
import typer
import wandb
from pathlib import Path
from data import corrupt_mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping

from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("WANDB_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 1) -> None:
    """Train a model on MNIST using PyTorch Lightning (train/val/test)."""
    print("Training with PyTorch Lightning")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    wandb.init(
        project="dtu_mlops",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    train_set, test_set = corrupt_mnist()
    # split a validation set from the train set (e.g., 10%)
    val_size = int(0.1 * len(train_set))
    train_size = len(train_set) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=5)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, num_workers=5)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=5)

    model = MyAwesomeModel()

    accelerator = (
        "gpu" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


    
    checkpoint_callbacks = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=PROJECT_ROOT / "lightning_logs" / "checkpoints",
        filename="mnist-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        mode="max",
    )

    early_stopping_callback = early_stopping.EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = Trainer(limit_train_batches=1.0,
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        default_root_dir=PROJECT_ROOT / "lightning_logs",
        callbacks=[checkpoint_callbacks, early_stopping_callback],
            logger=WandbLogger(project="dtu_mlops"))

    trainer.fit(model, train_dataloader, val_dataloader)

    print("Training complete — running test set")
    test_results = trainer.test(model, dataloaders=test_dataloader)
    print(f"Test results: {test_results}")

    # Create directories if they don't exist and save model state_dict
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")

    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
    )
    artifact.add_file(f"{model_dir / 'model.pth'}")
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    typer.run(train)
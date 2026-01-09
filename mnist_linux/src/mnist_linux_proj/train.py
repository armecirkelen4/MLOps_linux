import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path
from data import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    
    # Create directories if they don't exist
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir / "model.pth")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(statistics["train_loss"])
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[1].plot(statistics["train_accuracy"])
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Accuracy")
    fig.savefig(figures_dir / "training_statistics.png")


if __name__ == "__main__":
    typer.run(train)
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model


def get_train_loader(batch_size, use_augmentation):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader


def train(model_name, epochs, batch_size, learning_rate, use_augmentation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_train_loader(batch_size, use_augmentation)

    model = get_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training model: {model_name}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Augmentation: {use_augmentation}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                "loss": running_loss / total,
                "acc": 100.0 * correct / total
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100.0 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Train Accuracy: {epoch_accuracy:.2f}%"
        )

    Path("saved_models").mkdir(exist_ok=True)

    aug_name = "aug" if use_augmentation else "no_aug"
    save_path = f"saved_models/{model_name}_{aug_name}.pth"

    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MNIST neural network models.")

    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["mlp", "cnn", "transformer"],
        help="Choose model: mlp, cnn, transformer"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use image augmentation during training"
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_augmentation=args.augment
    )


if __name__ == "__main__":
    main()

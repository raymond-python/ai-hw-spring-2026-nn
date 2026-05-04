import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import get_model


def get_test_loader(batch_size):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


def test(model_name, batch_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = get_test_loader(batch_size)

    model = get_model(model_name).to(device)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train the model first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()

                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = 100.0 * correct / total

    print(f"Model: {model_name}")
    print(f"Test samples: {total}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    Path("results").mkdir(exist_ok=True)

    result_file = Path("results") / f"{model_name}_test_results.txt"

    with open(result_file, "w") as f:
        f.write("MNIST Test Results\n")
        f.write("==================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Test samples: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")

        f.write("Per-class Accuracy:\n")
        for digit in range(10):
            digit_acc = 100.0 * class_correct[digit] / class_total[digit]
            f.write(f"Digit {digit}: {digit_acc:.2f}%\n")

    print(f"Results saved to: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Test MNIST neural network models.")

    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["mlp", "cnn", "transformer"],
        help="Choose model: mlp, cnn, transformer"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model file"
    )

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"saved_models/{args.model}_no_aug.pth"

    test(
        model_name=args.model,
        batch_size=args.batch_size,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()

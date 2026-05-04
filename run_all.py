import os
import subprocess


commands = [
    "python train.py --model mlp --epochs 5",
    "python test.py --model mlp --model-path saved_models/mlp_no_aug.pth",

    "python train.py --model cnn --epochs 5",
    "python test.py --model cnn --model-path saved_models/cnn_no_aug.pth",

    "python train.py --model transformer --epochs 5",
    "python test.py --model transformer --model-path saved_models/transformer_no_aug.pth",

    "python train.py --model cnn --epochs 5 --augment",
    "python test.py --model cnn --model-path saved_models/cnn_aug.pth"
]


def main():
    for command in commands:
        print("=" * 80)
        print(f"Running: {command}")
        print("=" * 80)
        subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()

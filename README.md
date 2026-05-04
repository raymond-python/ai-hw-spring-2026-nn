# AI HW Spring 2026 NN

This project implements small image recognition neural networks using the MNIST handwritten digit dataset.

The project trains models on the MNIST training set and tests them on the MNIST test set.

## Dataset

MNIST contains handwritten digit images from 0 to 9.

- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28 x 28 grayscale

The dataset is loaded from `torchvision.datasets.MNIST`.

## Models

This project includes three models:

1. Shallow Multi-Layer Perceptron, also called MLP.
2. Convolutional Neural Network, also called CNN.
3. Transformer Encoder, similar to a tiny Vision Transformer.

## Project Structure

```text
ai-hw-spring-2026-nn/
│
├── README.md
├── requirements.txt
├── models.py
├── train.py
├── test.py
├── run_all.py
├── results/
└── saved_models/
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Train Models

Train MLP:

```bash
python train.py --model mlp --epochs 5
```

Train CNN:

```bash
python train.py --model cnn --epochs 5
```

Train Transformer Encoder:

```bash
python train.py --model transformer --epochs 5
```

Train CNN with image augmentation:

```bash
python train.py --model cnn --epochs 5 --augment
```

## Test Models

Test MLP:

```bash
python test.py --model mlp --model-path saved_models/mlp_no_aug.pth
```

Test CNN:

```bash
python test.py --model cnn --model-path saved_models/cnn_no_aug.pth
```

Test Transformer Encoder:

```bash
python test.py --model transformer --model-path saved_models/transformer_no_aug.pth
```

Test augmented CNN:

```bash
python test.py --model cnn --model-path saved_models/cnn_aug.pth
```

## Run Everything

```bash
python run_all.py
```

## Example Results

The actual results may be slightly different depending on computer, random initialization, and number of epochs.

| Model | Test Accuracy |
|---|---:|
| MLP | around 97% |
| CNN | around 98% to 99% |
| Transformer Encoder | around 96% to 98% |
| CNN with Augmentation | around 98% to 99% |

## Notes

The CNN usually performs the best because convolution layers are very useful for image data.

The MLP can still work well on MNIST, but it loses some spatial information because it flattens the image.

The Transformer Encoder model divides the image into patches and learns relationships between those patches.

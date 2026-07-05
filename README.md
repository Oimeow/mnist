# MNIST Digit Classifier from Scratch

***Written by Damian Toma (Oimeow). Strictly non-AI UGC.***

A feedforward neural network built with NumPy that classifies handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) without any deep learning frameworks.

## Architecture

```
Input (784) -> Dense (256) -> ReLU -> Dense (128) -> ReLU -> Dense (10) -> Softmax
```

The input is a flattened 28x28 grayscale image. The output is a probability distribution over the 10 digit classes (0-9).

## How It Works

- **Forward pass:** activations flow through layers via `A = XW + b`
- **Loss:** cross-entropy between predicted probabilities and one-hot labels
- **Backward pass:** gradients are backpropagated manually through each module
- **Update:** vanilla gradient descent (`W -= lr * dW`)

## Data

MNIST is downloaded automatically via `sklearn.datasets.fetch_openml`. It contains 70,000 labeled images, which I split into:

| Split | Size |
|-------|------|
| Train | 56,000 |
| Val   | 7,000 |
| Test  | 7,000 |

Pixel values are min-max normalized to [0, 1].

## Training

Default hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| Epochs         | 25    |
| Batch size     | 128   |
| Learning rate  | 0.01  |

## Requirements

```
numpy
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

Run the notebook top to bottom. It will:

1. Download and preprocess MNIST
2. Define and initialize the network
3. Train for 25 epochs (~3mins), printing loss and accuracy each epoch
4. Plot the training loss curve
5. Display a 10x5 grid of test predictions (green = correct, red = incorrect)

This notebook is alternatively available ([here](https://colab.research.google.com/drive/1ThTHrdhnc5xlQ12-ZU0K-9TrnUCU8JRm?usp=sharing)) on Google Colab

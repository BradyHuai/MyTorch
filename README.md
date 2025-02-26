# MyTorch

**MyTorch** is a lightweight, educational implementation of a deep learning framework inspired by PyTorch. It is designed to help users understand the core concepts of automatic differentiation, tensor operations, and neural network training from scratch.
* Docs: https://minitorch.github.io/
---

## Features

- **Tensor Operations**: Supports basic tensor operations like addition, multiplication, and matrix operations.
- **Autograd**: Implements automatic differentiation for computing gradients.
- **Neural Network Modules**: Provides building blocks for creating neural networks (e.g., layers, activations, loss functions).
- **Optimizers**: Includes common optimization algorithms like Stochastic Gradient Descent (SGD).
- **Customizable**: Easily extendable for experimenting with new layers, optimizers, or loss functions.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/BradyHuai/MyTorch.git
cd MyTorch
pip install -r requirements.txt
pip install -r requirements.extra.txt
python -m pip install -Ue .
```


## Tests:

```
pytest
```

## Run
MNIST is already provided in `project/data`,
```bash
python project/run_mnist_multiclass.py
python project/run_sentiment.py
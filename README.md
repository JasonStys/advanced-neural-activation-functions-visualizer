# Advanced Neural Activation Functions Visualizer

This repository contains a CS478 deep learning assignment notebook that implements and visualizes common and advanced neural network activation functions. The project uses Python, NumPy, TensorFlow, and Matplotlib to plot each activation function over an input range from -5 to 5.

## Project Overview

Activation functions are a core part of neural networks because they introduce nonlinear behavior into a model. This notebook compares classic activation functions such as Step, Sigmoid, Tanh, and ReLU with more advanced functions such as GeLU, Softplus, ELU, SELU, Leaky ReLU, PReLU, SiLU, and Gaussian.

The goal of the project is to show how each activation function transforms input values and to make the differences between threshold, saturating, rectified, smooth, and self normalizing functions easier to understand visually.

## Functions Included

### Step Function

The step function returns 0 for negative input values and 1 for nonnegative input values. It is a simple threshold based activation.

### Sigmoid

The sigmoid function maps values into the range from 0 to 1. It is often associated with probability style binary outputs.

### Tanh

The tanh function maps values into the range from -1 to 1. It is similar to sigmoid, but it is centered around 0.

### ReLU

The Rectified Linear Unit returns 0 for negative values and returns the input value for positive values. It is widely used because it is simple and efficient.

### GeLU

The Gaussian Error Linear Unit is a smoother activation function that uses the Gaussian error function. The notebook implements it with TensorFlow.

```python
def GeLU(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
```

### Softplus

Softplus is a smooth approximation of ReLU.

```python
def Softplus(x):
    return np.log(1 + np.exp(x))
```

### ELU

The Exponential Linear Unit returns the input for positive values and uses an exponential curve for negative values.

```python
def ELU(x):
    return np.where(x > 0, x, (np.exp(x) - 1))
```

### SELU

The Scaled Exponential Linear Unit scales the ELU style function to support self normalizing neural network behavior.

```python
def SELU(x):
    return np.where(x > 0, 1.05 * x, (1.67 * 1.05) * (np.exp(x) - 1))
```

### Leaky ReLU

Leaky ReLU keeps a small nonzero slope for negative values, which helps avoid completely inactive neurons.

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

### PReLU

Parametric ReLU is similar to Leaky ReLU, but the negative slope is treated as a tunable parameter.

```python
def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)
```

### SiLU

The Sigmoid Linear Unit multiplies the input by its sigmoid value, creating a smooth nonlinear activation.

```python
def silu(x):
    return x / (1 + np.exp(-x))
```

### Gaussian

The Gaussian function creates a bell shaped curve centered around 0.

```python
def gaussian(x):
    return np.exp(-(x ** 2))
```

## Technologies Used

- Python
- NumPy
- TensorFlow
- Matplotlib
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── CS_478_Assignment_1.ipynb
├── CS 478 Assignment #1.pdf
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/advanced-neural-activation-functions-visualizer.git
cd advanced-neural-activation-functions-visualizer
```

Install the required packages.

```bash
pip install numpy tensorflow matplotlib notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- Python programming
- NumPy based mathematical computation
- TensorFlow mathematical operations
- Neural network activation function implementation
- Matplotlib data visualization
- Comparing nonlinear function behavior
- Jupyter Notebook workflow
- Google Colab workflow

## Notes

The notebook includes both standard and advanced activation functions. The notebook comments identify some functions as collaborative work, with Leaky ReLU, PReLU, SiLU, and Gaussian listed as Jason Stys contributions.

The exported PDF includes most plotted outputs, but the last page cuts off part of the SiLU plot. The notebook contains the complete SiLU and Gaussian cells.

## Possible Future Improvements

- Add derivative plots for each activation function
- Add a single comparison plot with all activation functions
- Add short explanations of where each activation function is commonly used
- Add unit tests for each function implementation
- Convert the notebook into a reusable Python module
- Add interactive controls for alpha values in Leaky ReLU and PReLU

# Physics-Informed Neural Networks for Multi-Scale Multi-Frequency PDEs

This repository provides a comprehensive implementation of Physics-Informed Neural Networks (PINNs) tailored to solve complex partial differential equations (PDEs) characterized by multi-scale and multi-frequency behaviors. It contains two main applications: one utilizing adaptive weighting schemes to stabilize training across heterogeneous loss components, and another leveraging Fourier feature embeddings to address spectral bias and improve the learning of high-frequency components.

## Problem Overview

Traditional neural networks often struggle with PDEs involving rapidly oscillating or multi-frequency solutions due to **spectral bias**, which causes a preference for learning low-frequency features first. Additionally, imbalanced contributions in loss terms (e.g., data fidelity vs. physical residuals) can hinder convergence. These challenges are especially pronounced in noisy data environments or when labeled data is scarce.

## Methodology

### Fourier Feature Embedding

To mitigate spectral bias in learning high-frequency functions, Fourier embeddings are applied to the input domain:
- Inputs are mapped to a higher-dimensional space using sinusoidal transformations.
- Encodings with multiple frequency scales are concatenated to enable learning of both low and high-frequency components simultaneously.
- The resulting network efficiently reconstructs multi-scale functions with minimal training data and demonstrates robustness to noise.

### Benefits of Fourier Embedding

- **Enhanced frequency resolution:** Empirically and theoretically shown to enable learning of high-frequency components by transforming the Neural Tangent Kernel (NTK).
- **Faster convergence:** Enables more uniform learning dynamics across spectral components.
- **Robustness to noise:** Maintains high fidelity in function approximation despite training data corruption.
- **Multi-scale capacity:** Combined embeddings adapt to different frequency bands without manual tuning of frequency scales.

### Governing Equation Example:
$$
\nabla_{xx} u(x) = f(x), \quad u(0) = u(1) = 0, \quad u(x) = \sin(2\pi x) + 0.1 \sin(50\pi x)
$$

This PDE is representative of systems exhibiting fine-grained oscillatory behavior. The objective of the Neural Network is to learn the function $f(x)$ from partial, noisy data. Results demonstrate significant improvements in solution accuracy and convergence behavior when using the multi-scale Fourier embedding technique, especially in the presence of noise or when modeling fine-resolution physics.

## Code Structure

- `pinn_fourier_embedding.py`: Solves 1D multi-frequency PDE using Fourier feature embeddings.
- `plotting_functions.py`: Utility for 2D/3D visualizations of the predicted and reference solutions.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.11
- NumPy, Matplotlib, Scikit-learn
- `pyDOE`, `rff` (for Latin Hypercube Sampling and Fourier embeddings)

Install dependencies using:
```bash
pip install -r requirements.txt

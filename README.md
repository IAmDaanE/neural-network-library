# Custom Neural Network Library from Scratch

A lightweight, object-oriented Deep Learning library built entirely from scratch in Python using only **NumPy** and **Pygame**. 

This library was developed to understand the inner workings of backpropagation, matrix-driven neural layers, gradient descent, and real-time network visualization without relying on high-level frameworks like PyTorch or TensorFlow.

## 🚀 Features

- **Matrix-Driven Layers**: Fully connected layout using optimized NumPy `@` matrix multiplication for fast batch processing.
- **Dynamic Weight Initializers**: Built-in implementations for **He (Kaiming)** and **Xavier (Glorot)** initialization to ensure training stability and prevent exploding/vanishing gradients.
- **Flexible Activations**: Built-in support for `ReLU`, `Linear`, and `Sigmoid` activations along with their respective exact gradient maps.
- **Stable Loss Functions**: Features numerical adjustments like Logit-shifting and clipping to provide rock-solid `Softmax Cross-Entropy` and `MSE` calculations.
- **Real-Time Pygame Visualization**: Dynamic monitoring of nodes, active weights, and real-time loss tracking during training epochs.
- **Modular Learning Rate Decays**: Built-in learning rate schedulers including `Exponential Decay` and `Inverse Time Decay`.
---
## visualization
<img width="1892" height="981" alt="image" src="https://github.com/user-attachments/assets/7bfa06eb-cfa9-4582-850a-6be700d1a83e" />
---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE). Feel free to use it to learn, hack, and expand your understanding of Neural Networks!

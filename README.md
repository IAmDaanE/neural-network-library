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

## 🛠️ Library Architecture

The library is split into highly modular components mimicking modern deep learning frameworks:

```text
├── nnlib/
│   ├── __init__.py
│   ├── network.py          # Network, Layer, and Activations classes
│   ├── losses.py           # MSE and Softmax Cross-Entropy loss gradients
│   ├── initializers.py     # He and Xavier weight initialization
│   └── schedulers.py       # Learning rate decay methods
```

---

## 💻 Quick Start: Training on MNIST

Here is how you can build, train, and test a network to classify handwritten digits from the MNIST dataset using `nnlib`.

### 1. Build and Train the Network (`train.py`)

```python
import numpy as np
import nnlib as nn
from utils import load_mnist_csv, shuffle_dataset, get_exponential_decay

# 1. Initialize the Network with a combined Softmax Cross-Entropy Loss
network = nn.Network(nn.Losses.softmax_cross_entropy)

# 2. Add layers (Using He initialization for ReLU, Xavier for Linear)
network.add(nn.Layer(784, 128, nn.Activations.relu, nn.WeightInitializers.he))
network.add(nn.Layer(128, 128, nn.Activations.relu, nn.WeightInitializers.he))
network.add(nn.Layer(128, 10, nn.Activations.linear, nn.WeightInitializers.xavier))

# 3. Load MNIST data (Normalized to 0.0 - 1.0, Labels as One-Hot vectors)
images, labels = load_mnist_csv("../data/mnist_train.csv")

batch_size = 64
start_lr = 0.1

def run_training(epochs):
    global images, labels
    current_lr = start_lr
    
    for epoch in range(epochs):
        # Apply Learning Rate Decay
        current_lr = get_exponential_decay(start_lr, 0.96, epoch)
        images, labels = shuffle_dataset(images, labels)
        
        # Mini-batch Training Loop (Fully Vectorized)
        for i in range(0, images.shape[0], batch_size):
            images_batch = images[i : i + batch_size]
            labels_batch = labels[i : i + batch_size]
            
            # Forward pass
            prediction = network.forward(images_batch)
            
            # Backward pass & Update
            network.backward(prediction, labels_batch)
            network.update(current_lr)
            
        # Real-time Pygame visualization (Called once per epoch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {network.loss:.4f} | LR: {current_lr:.4f}")
            network.visualize(1280, 720)

if __name__ == "__main__":
    run_training(100)
```

### 2. Evaluate Model Accuracy (`test.py`)

Using vectorized NumPy comparisons, evaluating the full 10,000 image test set takes less than a second:

```python
import numpy as np
import nnlib as nn

# Load your test dataset and trained weights...
# (Assuming network weights are loaded into network.layers)

# Vectorized forward pass through the entire test set
all_predictions = network.forward(test_images) # Shape: (10000, 10)

# Extract predicted digits and actual digits
guessed_numbers = np.argmax(all_predictions, axis=1)
actual_numbers = np.argmax(test_labels, axis=1)

# Calculate exact accuracy percentage
accuracy = np.mean(guessed_numbers == actual_numbers) * 100
print(f"✅ Final Test Dataset Accuracy: {accuracy:.2f}%")
```

---

## 📈 Performance

When configured with a `784 -> 128 -> 128 -> 10` architecture utilizing **He Initialization**, **Exponential LR Decay**, and **Softmax Cross-Entropy**, this library reliably achieves:

- **Training Start Loss**: ~2.30 (Perfect mathematical verification of random guessing over 10 classes).
- **Final Test Accuracy**: **94% - 97%** within 30-50 epochs.

---

## 🧠 Math Behind the Slices

Unlike naive implementations that require `for`-loops for individual data samples, this library is completely optimized for **batch computation**. 

During the forward pass of a batch size of 64:
- The input matrix `(64, 784)` is multiplied by the weight matrix `(784, 128)`.
- NumPy uses optimized C-level BLAS libraries to compute the resulting `(64, 128)` hidden activations instantaneously.
- **Broadcasting** safely adds the bias row vector `(1, 128)` across all 64 individual samples simultaneously.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE). Feel free to use it to learn, hack, and expand your understanding of Neural Networks!

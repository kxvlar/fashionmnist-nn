# Fashion MNIST Embedding Learning Project

A deep learning project that trains a convolutional neural network (CNN) to learn well-separated embeddings for Fashion MNIST classification. The model uses center loss optimization to encourage samples from the same class to cluster together in a 3D embedding space.

## Overview

This project implements an advanced neural network architecture that combines standard classification loss with center loss to learn meaningful embeddings. The trained model achieves strong clustering performance with an Adjusted Rand Index (ARI) of 0.67, demonstrating effective separation of fashion item classes in the embedding space.

### Key Features

- **Advanced Architecture**: Multi-scale CNN with batch normalization and adaptive pooling
- **Center Loss Optimization**: Encourages class-specific clustering in embedding space
- **Efficient Models**: JIT-scripted models under 5 MB for deployment
- **3D Visualization**: Interactive Plotly visualization of learned embeddings
- **High-Quality Clustering**: Validated with Gaussian Mixture Models (ARI: 0.67)

## Dataset

**Fashion MNIST** - 28×28 grayscale images across 10 clothing classes:
- Ankle Boot
- Pullover
- Trouser
- Shirt
- Coat
- Sandal
- Sneaker
- Dress
- Bag
- T-shirt/top

- **Training Samples**: 60,000
- **Test Samples**: 10,000

## Model Architecture

```
SimpleCNN
├── Feature Extraction
│   ├── Conv2d(1→32) + BatchNorm + ReLU
│   ├── Conv2d(32→32) + BatchNorm + ReLU
│   ├── MaxPool2d(2)
│   ├── Conv2d(32→64) + BatchNorm + ReLU
│   ├── Conv2d(64→64) + BatchNorm + ReLU
│   ├── MaxPool2d(2)
│   ├── Conv2d(64→96) + BatchNorm + ReLU
│   └── AdaptiveAvgPool2d(1)
├── Embedding (96 → 64 → 3D)
│   ├── Linear(96→64) + ReLU
│   └── Linear(64→3)
└── Classification Head
    └── Linear(3→10)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | Cosine Annealing (T_max=5) |
| Batch Size | 64 |
| Epochs | 10 |
| Classification Loss | Cross-Entropy with label smoothing (0.02) |
| Center Loss Weight | 0.12 |
| Random Seed | 154 |

### Center Loss

The model is trained with a combined loss function:

```
L_total = L_classification + λ * L_center

where:
  L_center = ||embedding - target_center[label]||²
  λ = 0.12 (center loss weight)
```

Target centers are predefined in 3D space to maximize separation between classes:
- Classes 0-3: ±3.0 on different axes
- Classes 4-9: ±4.2 on single axes

## Results

### Performance Metrics

- **Test Accuracy**: ~91% (achieved during training)
- **Adjusted Rand Index (ARI)**: 0.67
  - Measures how well the learned embeddings separate classes
  - Computed using Gaussian Mixture Model clustering on embeddings

### Model Artifacts

| File | Size | Description |
|------|------|-------------|
| `kevton_nn_model.pt` | 0.53 MB | JIT-scripted trained model |
| `kevton_nn_model_1.pt` | 0.54 MB | Variant 1 checkpoint |
| `kevton_nn_model_2.pt` | 0.55 MB | Variant 2 checkpoint |
| `robust_center_nn_model.pt` | 0.53 MB | Robustness variant |

### Visualization

- `newplot.png`: Interactive 3D scatter plot showing all 10,000 test samples
  - X, Y, Z axes represent the 3D embedding space
  - Colors indicate Fashion MNIST class
  - Demonstrates clear clustering of similar items

## Project Structure

```
lab_11/
├── README.md                          # This file
├── challenge.ipynb                    # Main training and evaluation notebook
├── tutorial_pytorch.ipynb             # PyTorch learning tutorial
├── perturbed.ipynb                    # Perturbation analysis notebook
├── kevton_nn_model.pt                 # Trained model
├── kevton_nn_model_1.pt               # Model checkpoint 1
├── kevton_nn_model_2.pt               # Model checkpoint 2
├── robust_center_nn_model.pt          # Robustness variant
├── newplot.png                        # 3D embedding visualization
├── perturbed_train_compressed.pt      # Perturbed training data
├── data/                              # Fashion MNIST dataset
└── stat254-env/                       # Virtual environment (optional)
```

## Usage

### Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
scikit-learn
matplotlib
plotly
```

### Loading and Using the Model

```python
import torch

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load("kevton_nn_model.pt", map_location=device)
model.eval()

# Get embeddings for new images
with torch.no_grad():
    images = ...  # Your Fashion MNIST images (batch_size, 1, 28, 28)
    embeddings = model.embedding(images)  # Shape: (batch_size, 3)
    predictions = model(images)           # Shape: (batch_size, 10)
```

### Training from Scratch

Run the Jupyter notebook `challenge.ipynb` to train the model from scratch. The notebook handles:
- Dataset loading
- Model initialization
- Training loop with center loss
- Model saving
- Evaluation and visualization

## Key Insights

1. **Embedding Space**: The 3D embedding space effectively separates similar clothing items, with clear boundaries between different classes
2. **Center Loss**: The combination of classification and center loss produces embeddings that are both discriminative and geometrically organized
3. **Clustering Quality**: High ARI score (0.67) indicates that the learned representations are highly amenable to unsupervised clustering
4. **Model Efficiency**: JIT-scripted models are compact (<1 MB) while maintaining full functionality

## Future Improvements

- Explore higher-dimensional embedding spaces for better separation
- Implement metric learning approaches (e.g., triplet loss, contrastive loss)
- Fine-tune on related datasets (MNIST, CIFAR-10)
- Add adversarial robustness training
- Implement model distillation for further size reduction

## References

- Fashion MNIST: [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- Center Loss: [ArXiv](https://arxiv.org/abs/1612.02190)
- PyTorch Documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html)

## Author

Kevton

## License

This project is part of STAT 254 coursework.

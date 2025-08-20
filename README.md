# Behavior Cloning Tutorial: Autonomous Driving with CNNs

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.21.0-green.svg)](https://gym.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

This repository provides a comprehensive tutorial on implementing **Behavior Cloning (BC)** for autonomous driving using Convolutional Neural Networks (CNNs). The tutorial demonstrates end-to-end imitation learning in the CarRacing-v0 environment from OpenAI Gym.

## Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Environment Setup](#environment-setup)
- [Data Collection](#data-collection)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Project Structure](#project-structure)
- [Advanced Configuration](#advanced-configuration)
- [References](#references)
- [Contributing](#contributing)

## Overview

**Behavior Cloning** is a supervised learning approach to imitation learning where a neural network learns to replicate expert behavior from demonstration data. This implementation focuses on autonomous driving, where:

- **Input**: RGB images from the simulation environment
- **Output**: Continuous control actions `[steering, brake, throttle]`
- **Learning Paradigm**: Supervised learning with expert trajectories

### Key Features

- CNN-based architecture for visual perception
- Temporal frame stacking for dynamic understanding  
- Expert data collection interface
- Comprehensive training and evaluation pipeline
- Performance visualization tools

## Technical Architecture

### Neural Network Design

The model implements a Convolutional Neural Network specifically designed for visual control tasks, based on the architecture proposed by [Irving et al. (2023)](https://repositorio.ufsc.br/handle/123456789/251825).

```
Input: 84x84x4 (grayscale, 4-frame stack)
    ↓
Conv2D Layers + Batch Normalization + ReLU
    ↓
Global Average Pooling
    ↓
Fully Connected Layers 
    ↓
Output: 3 continuous actions [steering, brake, throttle]
```

<p align="center">
  <img src="carRacing_CNN.png" alt="CNN Architecture" width="600"/>
  <br>
  <em>Figure 1: CNN Architecture for Behavior Cloning</em>
</p>

### Data Preprocessing Pipeline

1. **Image Preprocessing**:
   - Convert RGB frames to grayscale (96x96) -> (84x84)
   - Normalize pixel values to [0,1]
   - Apply temporal stacking (4 consecutive frames)

2. **Action Space**:
   - Steering: [-1, 1] (left/right)
   - Brake: [0, 1] (no brake/full brake)
   - Throttle: [0, 1] (no gas/full gas)

## Environment Setup

### Prerequisites

- Python 3.8+

### Quick Start

Navigate to the `src` directory and execute:

```bash
make run
```

This command will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Execute the training (`main.py`)
4. Generate evaluation metrics and visualizations

### Manual Installation

```bash
# Create virtual environment
python -m venv bc_env
source bc_env/bin/activate  # Linux/Mac
# bc_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Collection

### Expert Data Generation

To collect expert demonstration data:

```bash
make expert
```

This launches an interactive session where:
- **Arrow Keys**: Control steering
- **Up Arrow**: Accelerate
- **Down Arrow**: Brake
- **ESC**: Exit and save trajectory

**Data Storage**: Trajectories are saved as pickle files in `src/data/trajectories/` containing observation-action pairs.

### Training Process

1. Load expert trajectories from `src/data/trajectories/`
2. Split data into train/validation sets (80/20)
3. Train CNN with Adam optimizer
4. Save best model based on validation loss
5. Generate performance metrics

## Evaluation and Visualization

### Performance Metrics

```bash
make plot
```

Generates visualizations for:
- Reward progression over episodes.

## Project Structure

```
simple-bc-tutorial/
├── src/
│   ├── main.py              # Main training script
│   ├── cnn.py               # CNN architecture definition
│   ├── car_racing_v0.py     # Expert data collection
│   ├── data/
│   │   └── trajectories/    # Expert demonstration data
│   ├── models/              # Saved model checkpoints
│   └── plots/               # Generated visualizations
├── Makefile                 # Build automation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Known Limitations

- **Distribution Shift**: Performance degrades when encountering states not in training data
- **Compounding Errors**: Small prediction errors can accumulate over time
- **Data Efficiency**: Requires substantial expert demonstration data

## Advanced Configuration

### Network Architecture Modifications

Edit `cnn.py` to experiment with:
- Different layer configurations
- Alternative activation functions
- Regularization techniques
- Attention mechanisms

### Training Enhancements

Modify `main.py` for:
- Data augmentation strategies
- Advanced optimizers (AdamW, RMSprop)
- Learning rate scheduling
- Early stopping criteria

### Environment Variations

The framework can be extended to other OpenAI Gym environments:
- LunarLander-v2
- BipedalWalker-v3
- Custom simulation environments

## References

### Core Publications

- Irving, B. (2023). [Imitation learning for autonomous driving: disagreement-regularization and behavior cloning with beta distribution](https://repositorio.ufsc.br/handle/123456789/251825). Master's Thesis, UFSC.

### Technical Resources

- [PyTorch Neural Network Tutorial](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)
- [Regression with Neural Networks](https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in)
- [PyTorch Complete Guide](https://www.guru99.com/pytorch-tutorial.html)
- [Interactive Colab Notebook](https://colab.research.google.com/drive/1IWRgLeTug-7NphtB54iDz8aJEi_OpWbQ?usp=sharing)

## Contributing

We welcome contributions to improve this tutorial! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)  
6. Open a Pull Request

---

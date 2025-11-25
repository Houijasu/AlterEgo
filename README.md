# AlterEgo

A hybrid CNN+GCN neural network for house price prediction, achieving **R² = 0.9711** with ensemble training.

## Overview

AlterEgo combines Convolutional Neural Networks (CNN) with Graph Convolutional Networks (GCN) to predict house prices. The model leverages both local feature extraction through CNNs and relational learning through graph neural networks.

## Features

- **Hybrid Architecture**: CNN branch for local feature extraction + GCN branch for graph-based relational learning
- **Advanced Feature Expansion**: RBF kernels, Fourier features, polynomial features, and piecewise linear features
- **Graph Construction**: k-NN and quantile-based hybrid graph structure
- **Stochastic Weight Averaging (SWA)**: Better generalization through weight averaging
- **Ensemble Training**: Train multiple models with different seeds for improved accuracy
- **Attention-based Fusion**: Multi-head attention to combine CNN and GCN outputs
- **Interactive CLI**: Beautiful terminal interface with Spectre.Console

## Requirements

- .NET 10.0+
- CUDA-capable GPU (optional, falls back to CPU)

## Installation

```bash
git clone https://github.com/Houijasu/AlterEgo.git
cd AlterEgo
dotnet restore
dotnet build
```

## Usage

### Interactive Mode

```bash
dotnet run
```

### Train Hybrid Model

```bash
# Basic training
dotnet run -- hybrid

# With custom parameters
dotnet run -- hybrid -e 500 --patience 100

# Ensemble training (recommended for best results)
dotnet run -- hybrid --ensemble 5 -e 500 --patience 80
```

### Make Predictions

```bash
# Using ML.NET model
dotnet run -- predict 2000

# Using Hybrid CNN+GCN model
dotnet run -- predict 2000 -m hybrid
```

### Run Benchmarks

```bash
dotnet run -- benchmark
```

## Architecture

```
Input (Size)
    │
    ▼
┌─────────────────────────────────────┐
│         Feature Expander            │
│  (RBF + Fourier + Polynomial)       │
└─────────────────────────────────────┘
    │                   │
    ▼                   ▼
┌─────────────┐   ┌─────────────┐
│ CNN Branch  │   │ GCN Branch  │
│ (Conv1D)    │   │ (GraphConv) │
└─────────────┘   └─────────────┘
    │                   │
    └───────┬───────────┘
            ▼
┌─────────────────────────────────────┐
│      Attention-based Fusion         │
│        (Multi-head Attention)       │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│          Output Head                │
│      (Price Prediction)             │
└─────────────────────────────────────┘
```

## Results

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.9597 |
| LightGBM | 0.9665 |
| Hybrid CNN+GCN (single) | 0.9673 |
| **Hybrid CNN+GCN (ensemble)** | **0.9711** |

## Configuration

Key hyperparameters in `HybridConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Epochs` | 2000 | Maximum training epochs |
| `LearningRate` | 5e-4 | Initial learning rate |
| `EarlyStopPatience` | 150 | Epochs without improvement before stopping |
| `GraphK` | 10 | k-NN neighbors for graph construction |
| `GnnHiddenDim` | 16 | GNN hidden dimension |
| `ExpandedFeatures` | 16 | Feature expansion output dimension |
| `UseSwa` | true | Enable Stochastic Weight Averaging |

## Project Structure

```
AlterEgo/
├── CLI/
│   └── Commands/           # CLI commands (hybrid, predict, benchmark)
├── Models/
│   └── Neural/             # Neural network configurations
├── Services/
│   ├── Neural/
│   │   ├── Layers/         # CNN, GCN, Attention layers
│   │   ├── GraphBuilder.cs # Graph construction
│   │   ├── HybridCnnGcnModel.cs
│   │   └── HybridTrainer.cs
│   ├── HybridModelService.cs
│   └── BenchmarkService.cs
└── Program.cs
```

## Technologies

- **TorchSharp**: PyTorch bindings for .NET
- **ML.NET**: Microsoft's machine learning framework
- **Spectre.Console**: Beautiful console UI

## License

MIT

## Acknowledgments

Built with Claude Code.

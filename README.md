# Malware Classification Using Neural Networks on Control Flow Graphs

## Overview
This repository focuses on **binary classification of malware vs benign executables** using Control Flow Graph (CFG) analysis combined with Neural Network architectures.

**Current Status**: Full GNN-based malware classification pipeline implemented!

## 🚀 Quick Start: Google Colab Notebook

**Want to run everything in one place?** Use our all-in-one Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bodiwael/CFG-E/blob/main/CFG_Malware_Classification_Colab.ipynb)

The notebook includes:
- ✅ Complete pipeline from binary files to trained model
- ✅ All functions integrated in one file
- ✅ Step-by-step execution with explanations
- ✅ Automatic GPU detection
- ✅ Interactive visualizations
- ✅ Easy file upload from Google Drive or local machine
- ✅ Download trained model and results

**Perfect for**: Learning, experimentation, and quick prototyping!

---

## Table of Contents
1. [Approach Comparison](#approach-comparison)
2. [Recommended: Graph Neural Networks (GNN)](#recommended-graph-neural-networks-gnn)
3. [Alternative: Abstract Interpretation](#alternative-abstract-interpretation)
4. [Other Approaches](#other-approaches)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Dataset Considerations](#dataset-considerations)
7. [Evaluation Metrics](#evaluation-metrics)

---

## Approach Comparison

### Summary Table

| Approach | Complexity | Model Fit | Interpretability | Performance | Training Time |
|----------|-----------|-----------|------------------|-------------|---------------|
| **GNN (Recommended)** | Medium | ⭐⭐⭐⭐⭐ Excellent | Medium | High | Medium |
| Abstract Interpretation | High | ⭐⭐ Poor | High | Unknown | Very High |
| CNN on Adjacency Matrix | Low-Medium | ⭐⭐⭐ Good | Low | Medium | Fast |
| Sequential (LSTM/RNN) | Medium | ⭐⭐⭐ Good | Low | Medium | Medium |
| Transformer | High | ⭐⭐⭐⭐ Very Good | Low | High | Slow |
| Traditional ML + Features | Low | ⭐⭐ Fair | High | Medium | Very Fast |

---

## Recommended: Graph Neural Networks (GNN)

### Why GNN? (Your Doctor is Right! 🎯)

**GNNs are the BEST fit for CFG-based malware classification** because:

1. **Natural Representation**: CFGs are graphs by nature
   - Nodes = basic blocks/functions
   - Edges = control flow/function calls
   - GNNs designed specifically for graph-structured data

2. **Captures Structural Patterns**:
   - Malware often has distinct control flow patterns (obfuscation, encryption loops, anti-debug checks)
   - GNNs learn these structural signatures

3. **State-of-the-Art Results**: Recent research shows 95%+ accuracy on malware detection

4. **Handles Variable Size**: Your CFG has 400+ nodes, others may have 50 or 5000 - GNNs handle this naturally

### Answer to Your Question: "Is Abstract Interpretation Too Advanced?"

**YES, for initial approach**. Here's why:
- Abstract interpretation generates semantic properties (value ranges, data dependencies)
- It's computationally expensive and complex to implement
- **The CFG structure itself contains rich information** - start here first!
- Your flowchart visualization (output.pdf) shows structural complexity is already present

### GNN Architecture Options

#### 1. **Graph Convolutional Networks (GCN)** ⭐ START HERE
```
Best for: Initial baseline, well-understood, fast training

Architecture:
Input CFG → GCN Layer 1 (128 units) → ReLU → Dropout(0.5)
         → GCN Layer 2 (64 units) → ReLU → Dropout(0.5)
         → Global Pooling → Dense(32) → Output(2 classes)

Node Features:
- Instruction count per basic block
- Opcode distribution (e.g., number of CALL, JMP, MOV instructions)
- String constants present
- API calls in block
```

#### 2. **Graph Attention Networks (GAT)**
```
Best for: Improved performance, learns which edges matter most

Similar architecture to GCN but with attention mechanisms
- Automatically learns important control flow paths
- Better for complex obfuscated malware
```

#### 3. **GraphSAGE**
```
Best for: Large-scale datasets, inductive learning

Can generalize to new nodes not seen during training
Good for evolving malware families
```

### Implementation Libraries

**PyTorch Geometric (PyG)** - Recommended
```python
# Install
pip install torch torch-geometric

# Basic GCN Example
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class MalwareGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(MalwareGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 2)  # Binary classification

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classifier
        x = self.fc(x)
        return x
```

**DGL (Deep Graph Library)** - Alternative
```python
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
# Good for very large graphs
```

### Data Preprocessing Pipeline for GNN

```python
# Step 1: Extract CFG (you already have this!)
import angr
import networkx as nx

def extract_cfg(binary_path):
    proj = angr.Project(binary_path, load_options={"auto_load_libs": False})
    cfg = proj.analyses.CFGFast(normalize=True)
    return cfg.graph

# Step 2: Convert to PyG format
from torch_geometric.utils import from_networkx

def cfg_to_pyg_data(cfg_graph, label):
    # Add node features
    for node in cfg_graph.nodes():
        cfg_graph.nodes[node]['x'] = extract_node_features(node)

    # Convert to PyG Data object
    data = from_networkx(cfg_graph)
    data.y = torch.tensor([label], dtype=torch.long)  # 0=benign, 1=malware
    return data

def extract_node_features(node):
    # Extract features from basic block
    # Examples:
    features = [
        node.size,  # Number of instructions
        count_calls(node),  # Number of CALL instructions
        count_jumps(node),  # Number of jumps
        has_strings(node),  # Contains string references
        # ... more features
    ]
    return torch.tensor(features, dtype=torch.float)
```

---

## Alternative: Abstract Interpretation

### What is Abstract Interpretation?

Abstract interpretation performs **static program analysis** to derive semantic properties:
- Value ranges of variables
- Possible data flows
- Invariants and loop bounds

### Why NOT Start with This?

**CONS:**
1. **Extremely Complex**: Requires implementing or using advanced static analysis frameworks (e.g., IKOS, Astrée)
2. **Computationally Expensive**: Can take minutes-hours per binary
3. **False Positives**: Abstract interpretation often over-approximates
4. **Dataset Size**: With VirusShare scale (millions of samples), this becomes impractical
5. **Overkill for Classification**: Structural patterns often sufficient

**PROS:**
1. **High Interpretability**: Know exactly what the analysis found
2. **Semantic Understanding**: Goes beyond syntax to behavior
3. **Research Value**: Novel if combined with deep learning

### When to Consider?

- **Phase 2 Project**: After baseline GNN proves effective
- **Feature Augmentation**: Use abstract interpretation to extract 10-20 high-level semantic features to add to GNN node features
- **Hybrid Approach**: GNN for structure + Abstract interpretation features for semantics

### Flowchart Visualization

You mentioned flowchart output (like your output.pdf) - this is already a visualization of CFG!
- ✅ Good for: Understanding individual programs, debugging
- ❌ Bad for: Direct input to neural networks (use graph structure instead)

---

## Other Approaches

### 3. CNN on Adjacency Matrix

**Concept**: Convert CFG to adjacency matrix, treat as image

```python
# CFG → NxN adjacency matrix → CNN

Architecture:
[N x N x 1] → Conv2D(32, 3x3) → MaxPool → Conv2D(64, 3x3) → MaxPool
            → Flatten → Dense(128) → Output(2)
```

**PROS:**
- Can use existing CNN architectures (ResNet, VGG)
- Fast training
- Proven for malware detection on raw bytes

**CONS:**
- ⚠️ Fixed size problem: Need to pad/truncate matrices
- Loses graph structure information
- Less interpretable than GNN
- Inferior to GNN for graph data

### 4. Sequential Models (LSTM/RNN)

**Concept**: Treat CFG as sequence of instruction opcodes

```python
# Instruction sequence → LSTM → Classification

[opcode_1, opcode_2, ..., opcode_n]
→ Embedding(256) → LSTM(128) → LSTM(64) → Dense(2)
```

**PROS:**
- Good for capturing execution patterns
- Works well with obfuscated code

**CONS:**
- Loses branching structure
- Limited context window
- Ordering ambiguity in CFG

### 5. Transformer Models

**Concept**: Self-attention on CFG nodes/instructions

**PROS:**
- Captures long-range dependencies
- State-of-the-art NLP techniques

**CONS:**
- Computationally expensive (O(n²) attention)
- Your CFG has 400+ nodes → expensive
- Harder to train, needs more data
- Overkill for this task

### 6. Traditional ML + Engineered Features

**Concept**: Extract graph metrics, use Random Forest/SVM

```python
Features:
- Number of nodes/edges
- Graph density
- Average/max degree
- Number of loops
- Betweenness centrality
- PageRank scores
- Function call graph depth
```

**PROS:**
- Fast to train
- Interpretable
- Good baseline

**CONS:**
- Manual feature engineering
- Lower accuracy than deep learning
- Doesn't scale to complex patterns

---

## Implementation Roadmap

### Phase 1: Baseline GNN (Weeks 1-3) ⭐ START HERE

1. **Data Collection** (Week 1)
   ```bash
   # Benign samples
   - Collect from: /usr/bin/, GitHub releases, clean software repos
   - Target: 1000 samples

   # Malware samples
   - Use VirusShare dataset you mentioned
   - Target: 1000 samples
   - ⚠️ Handle in isolated environment!
   ```

2. **Feature Extraction** (Week 1-2)
   ```python
   # Extend cfg_gen.py
   - Extract CFGs for all samples
   - Save as GraphML or PyG format
   - Extract basic node features (5-10 features per node)
   ```

3. **Model Training** (Week 2-3)
   ```python
   # Implement basic GCN
   - Train/val/test split: 70/15/15
   - Start with simple 2-layer GCN
   - Optimize hyperparameters
   ```

4. **Evaluation** (Week 3)
   - Accuracy, Precision, Recall, F1
   - Confusion matrix
   - False positive analysis

### Phase 2: Improved GNN (Weeks 4-6)

1. **Richer Node Features**
   - Add opcode n-grams
   - API call sequences
   - Register usage patterns
   - Constant pool features

2. **Architecture Improvements**
   - Try GAT (attention)
   - Deeper networks (3-4 layers)
   - Skip connections
   - Different pooling strategies

3. **Handle Imbalanced Data**
   - Class weighting
   - Oversampling minority class
   - Focal loss

### Phase 3: Advanced (Weeks 7+)

1. **Multi-class Classification**
   - Malware family classification
   - Ransomware, Trojan, Worm, etc.

2. **Explainability**
   - GNNExplainer: Which subgraphs indicate malware?
   - Attention visualization

3. **Consider Abstract Interpretation**
   - Add semantic features
   - Hybrid GNN + symbolic analysis

---

## Dataset Considerations

### VirusShare Dataset

**Links you provided:**
- https://virusshare.com/file?a18ef929b527a856737b9eb013f809ceae2b3059cf4b108c783f79e07849bca3
- https://virusshare.com/torrents

**Important Notes:**

⚠️ **SAFETY FIRST**
```bash
# Always work in isolated environment
- Use VM (VirtualBox/VMware)
- No network access during analysis
- Separate storage
- Regular snapshots
```

**Dataset Structure:**
```
dataset/
├── benign/
│   ├── sample_0001.exe
│   ├── sample_0002.exe
│   └── ...
├── malware/
│   ├── sample_0001.exe
│   ├── sample_0002.exe
│   └── ...
└── processed/
    ├── benign/
    │   ├── sample_0001.pt  # PyG Data object
    │   └── ...
    └── malware/
        ├── sample_0001.pt
        └── ...
```

### Benign Sources

1. **System Binaries**
   ```bash
   /usr/bin/*
   /bin/*
   C:\Windows\System32\*.exe
   ```

2. **Popular Open Source**
   - Python, Node.js, Git binaries
   - Firefox, Chrome installers
   - Common utilities (7zip, VLC, etc.)

3. **Dataset Repositories**
   - https://github.com/angr/binaries
   - VirusTotal benign corpus

### Recommended Sizes

**For Research/Learning:**
- Start: 500 benign + 500 malware
- Target: 2,000 benign + 2,000 malware

**For Production:**
- Minimum: 10,000 benign + 10,000 malware
- Optimal: 50,000+ each class

---

## Evaluation Metrics

### Primary Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# Key metrics for malware detection
1. Accuracy: Overall correctness
2. Precision: Of flagged malware, how many are actually malware?
3. Recall: Of actual malware, how many did we catch?
4. F1-Score: Harmonic mean of precision/recall

# For malware detection, RECALL is critical!
# Missing malware (false negative) is worse than false alarm
```

### Confusion Matrix

```
                Predicted
              Benign  Malware
Actual Benign   TN      FP     <- Minimize FP (false alarms)
       Malware  FN      TP     <- MINIMIZE FN (missed malware!)
```

### ROC Curve & AUC

```python
from sklearn.metrics import roc_curve, auc

# Plot ROC curve to choose optimal threshold
# AUC closer to 1.0 = better model
```

---

## Quick Start Code

### Complete Pipeline Example

```python
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import angr
import networkx as nx
import numpy as np

# 1. Extract CFG and convert to PyG Data
def process_binary(binary_path, label):
    """
    Args:
        binary_path: Path to executable
        label: 0 for benign, 1 for malware
    Returns:
        PyG Data object
    """
    # Extract CFG using angr
    proj = angr.Project(binary_path, load_options={"auto_load_libs": False})
    cfg = proj.analyses.CFGFast(normalize=True)
    G = cfg.graph

    # Convert to directed graph with integer node IDs
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Extract node features (example: simple features)
    node_features = []
    for node_id in G.nodes():
        original_node = list(mapping.keys())[node_id]
        features = [
            original_node.size if hasattr(original_node, 'size') else 1,
            G.in_degree(node_id),
            G.out_degree(node_id),
        ]
        node_features.append(features)

    # Create PyG Data
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# 2. Define GNN Model
class MalwareDetectorGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classification
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)

# 3. Training Loop
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# 4. Evaluation
def evaluate(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()

    return correct / len(loader.dataset)

# 5. Main execution
if __name__ == "__main__":
    # Process dataset
    dataset = []

    # Add benign samples
    for binary in benign_binaries:
        dataset.append(process_binary(binary, label=0))

    # Add malware samples
    for binary in malware_binaries:
        dataset.append(process_binary(binary, label=1))

    # Train/test split
    from torch_geometric.data import random_split
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MalwareDetectorGNN(num_features=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(100):
        loss = train_model(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}')
```

---

## Recommended Reading

### Papers on GNN for Malware Detection

1. **"Malware Detection on Byte Streams of PDF Files Using Convolutional Neural Networks"** (2018)
2. **"Deep Learning for Classification of Malware System Call Sequences"** (2016)
3. **"Graph Convolutional Networks for Malware Detection"** (2020)
4. **"Hierarchical Graph Convolutional Networks for Malware Family Classification"** (2021)

### GNN Tutorials

1. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/
2. Stanford CS224W: Machine Learning with Graphs
3. DGL Tutorials: https://docs.dgl.ai/tutorials/blitz/index.html

---

## Final Recommendation

### 🎯 **Start with GNN (Graph Convolutional Networks)**

**Rationale:**
1. ✅ Perfect fit for your CFG data
2. ✅ Your doctor's recommendation is spot-on
3. ✅ State-of-the-art performance
4. ✅ Reasonable complexity for research project
5. ✅ Plenty of open-source tools (PyTorch Geometric)

**Skip Abstract Interpretation for now** - it's overkill and won't improve your model significantly for initial classification task.

### Implementation Timeline

**Week 1-2:** Data collection + CFG extraction (extend cfg_gen.py)
**Week 3-4:** Implement basic 2-layer GCN with simple node features
**Week 5-6:** Evaluate, tune hyperparameters, improve features
**Week 7+:** Advanced architectures (GAT), explainability, multi-class

---

## Questions?

Feel free to ask about:
- Specific GNN architecture details
- Feature engineering for CFG nodes
- Dataset preparation
- Handling large-scale graphs
- Training optimization

Good luck with your research! 🚀🔬

---

**Project Structure Recommendation:**
```
CFG-E/
├── README.md (this file)
├── cfg_gen.py (your current extractor)
├── data/
│   ├── benign/
│   └── malware/
├── src/
│   ├── extract_features.py
│   ├── models/
│   │   ├── gcn.py
│   │   ├── gat.py
│   │   └── graphsage.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── visualization.ipynb
├── configs/
│   └── config.yaml
└── requirements.txt
```

---

## Implementation Status ✅

**The GNN-based classifier has been implemented!** All implementation files are now available in the repository.

---

## Quick Start Guide

### Option 1: Google Colab (Recommended for Beginners)

**Easiest way to get started!** Open the [Colab notebook](https://colab.research.google.com/github/bodiwael/CFG-E/blob/main/CFG_Malware_Classification_Colab.ipynb) and run all cells. Everything is included in one file!

### Option 2: Local Installation

For advanced users who want to run on their own machine:

### 1. Installation

```bash
# Clone the repository (if needed)
git clone <repository-url>
cd CFG-E

# Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

```bash
# Place your binaries in the appropriate directories
# Benign executables → data/raw/benign/
# Malware executables → data/raw/malware/

# Example structure:
# data/raw/benign/program1.exe
# data/raw/benign/program2.exe
# data/raw/malware/malware1.exe
# data/raw/malware/malware2.exe
```

### 3. Extract CFGs from Binaries

```bash
# Extract Control Flow Graphs from all binaries
python src/extract_batch.py \
    --benign-dir data/raw/benign \
    --malware-dir data/raw/malware \
    --output-benign data/processed/benign \
    --output-malware data/processed/malware \
    --workers 4

# This will create GraphML files for each binary
```

### 4. Extract Features from CFGs

```bash
# Convert CFGs to PyTorch Geometric format with node features
python src/extract_features.py \
    --benign-cfg-dir data/processed/benign \
    --malware-cfg-dir data/processed/malware \
    --output-benign data/processed/benign \
    --output-malware data/processed/malware \
    --workers 4

# This creates .pt files (PyTorch Geometric Data objects)
```

### 5. Train the Model

```bash
# Train the GNN model
python src/train.py --config configs/config.yaml

# Training will:
# - Split data into train/val/test sets (70/15/15)
# - Train the GCN model
# - Save best model based on validation accuracy
# - Generate training history
# - Evaluate on test set

# Results will be saved to: results/run_TIMESTAMP/
```

### 6. Evaluate the Model

```bash
# Generate detailed evaluation metrics and plots
python src/evaluate.py --results-dir results/run_TIMESTAMP/

# This generates:
# - Classification report (precision, recall, F1)
# - Confusion matrix (with visualization)
# - ROC curve and AUC score
# - Precision-Recall curve
# - Training history plots
```

### 7. Test the Dataset Loader (Optional)

```bash
# Test if your dataset is loading correctly
python src/dataset.py \
    --benign-dir data/processed/benign \
    --malware-dir data/processed/malware \
    --batch-size 32
```

---

## Configuration

Edit `configs/config.yaml` to customize training:

```yaml
# Model architecture
model:
  type: "gcn"  # Options: gcn, gcn_deep, gat, graphsage
  hidden_channels: 64  # Increase for more capacity
  dropout: 0.5  # Increase if overfitting

# Training parameters
training:
  epochs: 200
  batch_size: 32  # Adjust based on GPU memory
  learning_rate: 0.001
  use_class_weights: true  # For imbalanced datasets
```

---

## Expected Results

With a properly balanced dataset (1000+ samples each class), you should expect:

- **Accuracy**: 90-95%
- **Precision**: 88-94%
- **Recall**: 90-96%
- **F1-Score**: 89-95%
- **ROC AUC**: 0.95+

Results depend on:
- Dataset size and quality
- Balance between benign/malware samples
- Diversity of malware families
- Feature richness

---

## Troubleshooting

### Issue: CFG extraction fails for some binaries

```bash
# This is normal - some binaries may be packed or corrupted
# Check data/extraction_metadata.json for success/failure stats
```

### Issue: Out of memory during training

```bash
# Reduce batch size in configs/config.yaml
batch_size: 16  # or even 8
```

### Issue: Model overfitting (high train acc, low val acc)

```bash
# Increase dropout or add regularization
dropout: 0.6  # or 0.7
weight_decay: 0.001  # increase L2 regularization
```

### Issue: Low accuracy

```bash
# Possible causes:
# 1. Dataset too small → collect more samples
# 2. Dataset imbalanced → ensure use_class_weights: true
# 3. Features not informative → add more node features
# 4. Model too simple → try model: type: "gcn_deep" or "gat"
```

---

## Advanced Usage

### Experiment with Different Models

```bash
# Try Graph Attention Network (GAT)
# Edit configs/config.yaml:
model:
  type: "gat"
  hidden_channels: 64

# Try deeper network
model:
  type: "gcn_deep"
  hidden_channels: 128
```

### Extract Single Binary CFG

```bash
# Use the original cfg_gen.py for single binaries
python cfg_gen.py  # Edit main.exe path inside
```

### Custom Node Features

Edit `src/extract_features.py` in the `extract_node_features()` function to add:
- Opcode n-grams
- API call patterns
- String constants
- Register usage
- Memory access patterns

---

## Project Structure

```
CFG-E/
├── README.md                              # This file
├── CFG_Malware_Classification_Colab.ipynb # 🚀 All-in-one Colab notebook
├── requirements.txt                       # Python dependencies
├── cfg_gen.py                             # Original single-binary CFG extractor
├── output.pdf                             # Example CFG visualization
├── static_cfg.dot                         # Example CFG (DOT format)
├── static_cfg.graphml                     # Example CFG (GraphML format)
├── data/                       # Dataset directory
│   ├── raw/
│   │   ├── benign/            # Place benign executables here
│   │   └── malware/           # Place malware executables here
│   └── processed/
│       ├── benign/            # Processed CFG files
│       └── malware/           # Processed CFG files
├── src/                        # Source code
│   ├── extract_batch.py       # Batch CFG extraction
│   ├── extract_features.py    # Feature extraction
│   ├── dataset.py             # Dataset loader
│   ├── model.py               # GNN model architectures
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── configs/
│   └── config.yaml            # Training configuration
└── results/                    # Training results (auto-generated)
    └── run_TIMESTAMP/
        ├── best_model.pt      # Best model checkpoint
        ├── config.yaml        # Config used for this run
        ├── history.json       # Training history
        ├── test_results.json  # Test set results
        └── *.png              # Plots and visualizations
```

---

## Next Steps for Improvement

Once you have a working baseline model:

1. **Richer Node Features**
   - Add opcode n-grams
   - Extract API call sequences
   - Include constant pool features

2. **Try Advanced Architectures**
   - Graph Attention Networks (GAT)
   - Deeper networks (4-6 layers)
   - Skip connections

3. **Multi-Class Classification**
   - Classify by malware family
   - Detect specific malware types

4. **Explainability**
   - Use GNNExplainer
   - Visualize attention weights
   - Identify suspicious subgraphs

5. **Consider Abstract Interpretation** (Advanced)
   - Add semantic features
   - Hybrid GNN + symbolic analysis

---

## Safety Warning ⚠️

When working with malware samples:

```bash
# ALWAYS use isolated environment
- Work in a VM (VirtualBox/VMware)
- Disable network access
- Use separate storage
- Take regular snapshots
- Never execute malware on host system
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{cfg-malware-gnn,
  title={Malware Classification Using Graph Neural Networks on Control Flow Graphs},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/CFG-E}
}
```

---

## License

[Specify your license here]

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Malware Hunting! 🔍🛡️**

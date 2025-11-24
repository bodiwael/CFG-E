# Malware Classification Using Neural Networks on Control Flow Graphs

## Overview
This repository focuses on **binary classification of malware vs benign executables** using Control Flow Graph (CFG) analysis combined with Neural Network architectures.

**Current Status**: CFG extraction implemented using `angr` (see `cfg_gen.py`)

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

**Next Step:** Would you like me to help implement the GNN-based classifier?

#!/usr/bin/env python3
"""
Graph Neural Network Models for Malware Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool


class MalwareGCN(nn.Module):
    """
    Graph Convolutional Network for Malware Classification

    Architecture:
        Input → GCN Layer 1 → ReLU → Dropout →
        GCN Layer 2 → ReLU → Dropout →
        Global Pooling → FC Layer → Output

    Args:
        num_node_features: Number of features per node
        hidden_channels: Number of hidden units
        num_classes: Number of output classes (2 for binary)
        dropout: Dropout rate
        pooling: Pooling method ('mean' or 'max')
    """

    def __init__(self, num_node_features, hidden_channels=64, num_classes=2,
                 dropout=0.5, pooling='mean'):
        super(MalwareGCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)

        # Classification layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


class MalwareGCNDeep(nn.Module):
    """
    Deeper Graph Convolutional Network with 4 layers

    Args:
        num_node_features: Number of features per node
        hidden_channels: Number of hidden units
        num_classes: Number of output classes
        dropout: Dropout rate
        pooling: Pooling method ('mean' or 'max')
    """

    def __init__(self, num_node_features, hidden_channels=128, num_classes=2,
                 dropout=0.5, pooling='mean'):
        super(MalwareGCNDeep, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        self.conv4 = GCNConv(hidden_channels // 2, hidden_channels // 2)

        self.fc1 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc2 = nn.Linear(hidden_channels // 4, num_classes)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)

        # Classification layers
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class MalwareGAT(nn.Module):
    """
    Graph Attention Network for Malware Classification

    Uses attention mechanisms to learn important edges

    Args:
        num_node_features: Number of features per node
        hidden_channels: Number of hidden units
        num_classes: Number of output classes
        heads: Number of attention heads
        dropout: Dropout rate
        pooling: Pooling method
    """

    def __init__(self, num_node_features, hidden_channels=64, num_classes=2,
                 heads=4, dropout=0.5, pooling='mean'):
        super(MalwareGAT, self).__init__()

        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)

        self.fc = nn.Linear(hidden_channels, num_classes)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)

        # Classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


class MalwareGraphSAGE(nn.Module):
    """
    GraphSAGE for Malware Classification

    Good for large-scale graphs and inductive learning

    Args:
        num_node_features: Number of features per node
        hidden_channels: Number of hidden units
        num_classes: Number of output classes
        dropout: Dropout rate
        pooling: Pooling method
    """

    def __init__(self, num_node_features, hidden_channels=64, num_classes=2,
                 dropout=0.5, pooling='mean'):
        super(MalwareGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(hidden_channels, num_classes)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GraphSAGE layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)

        # Classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


def create_model(model_type='gcn', num_node_features=10, hidden_channels=64,
                num_classes=2, dropout=0.5, pooling='mean', **kwargs):
    """
    Factory function to create models

    Args:
        model_type: Type of model ('gcn', 'gcn_deep', 'gat', 'graphsage')
        num_node_features: Number of input features
        hidden_channels: Hidden layer size
        num_classes: Number of output classes
        dropout: Dropout rate
        pooling: Pooling method
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The created model
    """
    if model_type == 'gcn':
        return MalwareGCN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling
        )
    elif model_type == 'gcn_deep':
        return MalwareGCNDeep(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling
        )
    elif model_type == 'gat':
        heads = kwargs.get('heads', 4)
        return MalwareGAT(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout,
            pooling=pooling
        )
    elif model_type == 'graphsage':
        return MalwareGraphSAGE(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Test model creation"""
    print("Testing model architectures...\n")

    # Create a dummy batch
    from torch_geometric.data import Data, Batch

    data1 = Data(
        x=torch.randn(10, 10),  # 10 nodes, 10 features
        edge_index=torch.randint(0, 10, (2, 20)),  # 20 edges
        y=torch.tensor([0])
    )

    data2 = Data(
        x=torch.randn(15, 10),  # 15 nodes, 10 features
        edge_index=torch.randint(0, 15, (2, 30)),  # 30 edges
        y=torch.tensor([1])
    )

    batch = Batch.from_data_list([data1, data2])

    # Test each model
    models = ['gcn', 'gcn_deep', 'gat', 'graphsage']

    for model_type in models:
        print(f"Testing {model_type.upper()}:")
        model = create_model(model_type=model_type, num_node_features=10)
        model.eval()

        output = model(batch)
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  Predictions: {output.argmax(dim=1).tolist()}")
        print()


if __name__ == '__main__':
    main()

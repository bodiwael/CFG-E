#!/usr/bin/env python3
"""
Feature Extraction from CFGs
Converts GraphML CFGs to PyTorch Geometric Data objects with node features
"""

import torch
import networkx as nx
import os
import json
import logging
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_node_features(node_id, graph, node_attrs):
    """
    Extract features for a single node (basic block)

    Args:
        node_id: Node identifier
        graph: NetworkX graph
        node_attrs: Node attributes dictionary

    Returns:
        list: Feature vector for the node
    """
    features = []

    # Feature 1: Node size (instruction count)
    # The node ID often contains size information like "CFGNode name [size]"
    size = 1
    if isinstance(node_id, str) and '[' in node_id and ']' in node_id:
        try:
            size = int(node_id.split('[')[-1].split(']')[0])
        except:
            size = 1
    features.append(float(size))

    # Feature 2: In-degree (number of incoming edges)
    in_degree = graph.in_degree(node_id)
    features.append(float(in_degree))

    # Feature 3: Out-degree (number of outgoing edges)
    out_degree = graph.out_degree(node_id)
    features.append(float(out_degree))

    # Feature 4: Is entry node (in-degree == 0)
    is_entry = 1.0 if in_degree == 0 else 0.0
    features.append(is_entry)

    # Feature 5: Is exit node (out-degree == 0)
    is_exit = 1.0 if out_degree == 0 else 0.0
    features.append(is_exit)

    # Feature 6: Betweenness centrality indicator (high traffic node)
    # Simplified: nodes with both high in and out degree
    is_hub = 1.0 if (in_degree > 2 and out_degree > 2) else 0.0
    features.append(is_hub)

    # Feature 7: Degree ratio (out_degree / (in_degree + 1))
    # Indicates branching behavior
    degree_ratio = float(out_degree) / (float(in_degree) + 1.0)
    features.append(degree_ratio)

    # Feature 8: Is branching node (out_degree > 1)
    is_branch = 1.0 if out_degree > 1 else 0.0
    features.append(is_branch)

    # Feature 9: Is merge node (in_degree > 1)
    is_merge = 1.0 if in_degree > 1 else 0.0
    features.append(is_merge)

    # Feature 10: Log of size (helps with large variations)
    import math
    log_size = math.log(size + 1)
    features.append(log_size)

    return features


def compute_graph_statistics(graph):
    """
    Compute graph-level statistics

    Args:
        graph: NetworkX graph

    Returns:
        dict: Graph statistics
    """
    try:
        # Basic statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # Avoid division by zero
        if num_nodes == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'avg_degree': 0.0,
                'density': 0.0,
                'is_connected': False
            }

        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        # Graph density
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0

        # Connectivity
        is_connected = nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph)

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density': density,
            'is_connected': is_connected
        }
    except Exception as e:
        logger.warning(f"Error computing graph statistics: {e}")
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'avg_degree': 0.0,
            'density': 0.0,
            'is_connected': False
        }


def cfg_to_pyg_data(cfg_path, label):
    """
    Convert CFG GraphML file to PyTorch Geometric Data object

    Args:
        cfg_path: Path to GraphML file
        label: 0 for benign, 1 for malware

    Returns:
        Data: PyTorch Geometric Data object or None if failed
    """
    try:
        # Load graph
        G = nx.read_graphml(cfg_path)

        # Skip empty graphs
        if G.number_of_nodes() == 0:
            logger.warning(f"Empty graph: {cfg_path}")
            return None

        # Create integer node mapping (required for PyG)
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Relabel nodes with integers
        G = nx.relabel_nodes(G, node_to_idx)

        # Extract node features
        node_features = []
        for node_id in range(len(node_list)):
            original_node = node_list[node_id]
            attrs = G.nodes[node_id]
            features = extract_node_features(original_node, G, attrs)
            node_features.append(features)

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)

        # Create edge index
        edge_list = list(G.edges())
        if len(edge_list) == 0:
            # Handle graphs with no edges (single node)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Create label
        y = torch.tensor([label], dtype=torch.long)

        # Compute graph statistics
        stats = compute_graph_statistics(G)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # Store additional metadata
        data.num_nodes = len(node_list)
        data.graph_stats = stats

        return data

    except Exception as e:
        logger.error(f"Error processing {cfg_path}: {e}")
        return None


def process_cfg_file(args):
    """Wrapper for multiprocessing"""
    cfg_path, label = args
    return cfg_to_pyg_data(cfg_path, label)


def process_all_cfgs(cfg_dir, label, output_dir, num_workers=None):
    """
    Process all CFG files in a directory

    Args:
        cfg_dir: Directory containing CFG GraphML files
        label: 0 for benign, 1 for malware
        output_dir: Directory to save PyG Data objects
        num_workers: Number of parallel workers

    Returns:
        list: Successfully processed file metadata
    """
    # Find all GraphML files
    cfg_files = []
    for root, dirs, files in os.walk(cfg_dir):
        for file in files:
            if file.endswith('_cfg.graphml'):
                cfg_files.append(os.path.join(root, file))

    logger.info(f"Found {len(cfg_files)} CFG files in {cfg_dir}")

    if not cfg_files:
        logger.warning(f"No CFG files found in {cfg_dir}")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments
    args_list = [(cfg_path, label) for cfg_path in cfg_files]

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    logger.info(f"Processing with {num_workers} workers...")

    # Process files
    successful = 0
    failed = 0
    metadata = []

    label_str = 'benign' if label == 0 else 'malware'

    with Pool(num_workers) as pool:
        with tqdm(total=len(args_list), desc=f"Processing {label_str} CFGs") as pbar:
            for idx, (cfg_path, data) in enumerate(zip(cfg_files,
                                                       pool.imap(process_cfg_file, args_list))):
                if data is not None:
                    # Save as .pt file
                    file_hash = os.path.basename(cfg_path).replace('_cfg.graphml', '')
                    output_path = os.path.join(output_dir, f"{file_hash}.pt")
                    torch.save(data, output_path)

                    successful += 1
                    metadata.append({
                        'file_hash': file_hash,
                        'label': label,
                        'num_nodes': data.num_nodes,
                        'num_features': data.x.shape[1],
                        'output_path': output_path
                    })
                else:
                    failed += 1

                pbar.update(1)

    logger.info(f"\nProcessing complete for {label_str}:")
    logger.info(f"  ✓ Successful: {successful}/{len(cfg_files)}")
    logger.info(f"  ✗ Failed: {failed}/{len(cfg_files)}")

    return metadata


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract features from CFGs')
    parser.add_argument('--benign-cfg-dir', type=str, default='data/processed/benign',
                        help='Directory containing benign CFG files')
    parser.add_argument('--malware-cfg-dir', type=str, default='data/processed/malware',
                        help='Directory containing malware CFG files')
    parser.add_argument('--output-benign', type=str, default='data/processed/benign',
                        help='Output directory for benign PyG data')
    parser.add_argument('--output-malware', type=str, default='data/processed/malware',
                        help='Output directory for malware PyG data')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--metadata-file', type=str, default='data/features_metadata.json',
                        help='Output file for feature metadata')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Feature Extraction Tool")
    logger.info("=" * 60)

    all_metadata = {
        'benign': [],
        'malware': [],
        'num_features': 10,  # Update if you change the number of features
        'feature_names': [
            'node_size',
            'in_degree',
            'out_degree',
            'is_entry',
            'is_exit',
            'is_hub',
            'degree_ratio',
            'is_branch',
            'is_merge',
            'log_size'
        ]
    }

    # Process benign CFGs
    if os.path.exists(args.benign_cfg_dir):
        logger.info(f"\nProcessing benign CFGs from {args.benign_cfg_dir}")
        all_metadata['benign'] = process_all_cfgs(
            args.benign_cfg_dir,
            label=0,
            output_dir=args.output_benign,
            num_workers=args.workers
        )

    # Process malware CFGs
    if os.path.exists(args.malware_cfg_dir):
        logger.info(f"\nProcessing malware CFGs from {args.malware_cfg_dir}")
        all_metadata['malware'] = process_all_cfgs(
            args.malware_cfg_dir,
            label=1,
            output_dir=args.output_malware,
            num_workers=args.workers
        )

    # Save metadata
    os.makedirs(os.path.dirname(args.metadata_file), exist_ok=True)
    with open(args.metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"\nMetadata saved to {args.metadata_file}")
    logger.info("=" * 60)
    logger.info("Feature extraction complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

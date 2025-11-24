#!/usr/bin/env python3
"""
Batch CFG Extraction Script
Processes multiple binaries and extracts their Control Flow Graphs (CFGs)
"""

import angr
import networkx as nx
import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def strip_none_attributes(G):
    """Remove None attributes from graph (required for GraphML export)"""
    for node, attrs in list(G.nodes(data=True)):
        for k, v in list(attrs.items()):
            if v is None:
                del attrs[k]

    for u, v, attrs in list(G.edges(data=True)):
        for k, val in list(attrs.items()):
            if val is None:
                del attrs[k]


def extract_cfg(binary_path, output_dir, label):
    """
    Extract CFG from a single binary

    Args:
        binary_path: Path to the binary file
        output_dir: Directory to save the extracted CFG
        label: 'benign' or 'malware'

    Returns:
        dict: Metadata about the extraction
    """
    binary_name = os.path.basename(binary_path)
    file_hash = get_file_hash(binary_path)

    metadata = {
        'filename': binary_name,
        'file_hash': file_hash,
        'label': label,
        'status': 'failed',
        'error': None,
        'num_nodes': 0,
        'num_edges': 0,
        'num_functions': 0,
        'extraction_time': None
    }

    try:
        start_time = datetime.now()

        # Load binary with angr
        proj = angr.Project(
            binary_path,
            load_options={
                'auto_load_libs': False,
                'main_opts': {
                    'backend': 'blob'
                }
            }
        )

        # Generate CFG
        cfg = proj.analyses.CFGFast(
            normalize=True,
            data_references=False,  # Faster without data refs
            cross_references=False  # Faster without cross refs
        )

        # Get the graph
        G = cfg.graph

        # Strip None attributes
        strip_none_attributes(G)

        # Create output filename based on hash
        output_base = os.path.join(output_dir, file_hash)

        # Export CFG graph
        nx.write_graphml(G, f"{output_base}_cfg.graphml")

        # Export call graph
        if cfg.kb.callgraph:
            strip_none_attributes(cfg.kb.callgraph)
            nx.write_graphml(cfg.kb.callgraph, f"{output_base}_callgraph.graphml")

        # Update metadata
        end_time = datetime.now()
        metadata['status'] = 'success'
        metadata['num_nodes'] = G.number_of_nodes()
        metadata['num_edges'] = G.number_of_edges()
        metadata['num_functions'] = len(cfg.kb.functions)
        metadata['extraction_time'] = (end_time - start_time).total_seconds()

        logger.info(f"✓ Extracted CFG from {binary_name}: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")

    except Exception as e:
        metadata['error'] = str(e)
        logger.error(f"✗ Failed to extract CFG from {binary_name}: {e}")

    return metadata


def extract_cfg_wrapper(args):
    """Wrapper function for multiprocessing"""
    return extract_cfg(*args)


def process_directory(input_dir, output_dir, label, num_workers=None):
    """
    Process all binaries in a directory

    Args:
        input_dir: Directory containing binaries
        output_dir: Directory to save CFGs
        label: 'benign' or 'malware'
        num_workers: Number of parallel workers (None = auto)

    Returns:
        list: Metadata for all processed files
    """
    # Get all files in input directory
    binary_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip very small files (likely not executables)
            if os.path.getsize(file_path) > 1024:
                binary_files.append(file_path)

    logger.info(f"Found {len(binary_files)} files in {input_dir}")

    if not binary_files:
        logger.warning(f"No files found in {input_dir}")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    args_list = [(binary_path, output_dir, label) for binary_path in binary_files]

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    logger.info(f"Processing with {num_workers} workers...")

    # Process files in parallel
    all_metadata = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(args_list), desc=f"Extracting {label} CFGs") as pbar:
            for metadata in pool.imap_unordered(extract_cfg_wrapper, args_list):
                all_metadata.append(metadata)
                pbar.update(1)

    # Calculate statistics
    successful = sum(1 for m in all_metadata if m['status'] == 'success')
    failed = len(all_metadata) - successful

    logger.info(f"\nProcessing complete for {label}:")
    logger.info(f"  ✓ Successful: {successful}/{len(all_metadata)}")
    logger.info(f"  ✗ Failed: {failed}/{len(all_metadata)}")

    return all_metadata


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Batch CFG extraction from binaries')
    parser.add_argument('--benign-dir', type=str, default='data/raw/benign',
                        help='Directory containing benign binaries')
    parser.add_argument('--malware-dir', type=str, default='data/raw/malware',
                        help='Directory containing malware binaries')
    parser.add_argument('--output-benign', type=str, default='data/processed/benign',
                        help='Output directory for benign CFGs')
    parser.add_argument('--output-malware', type=str, default='data/processed/malware',
                        help='Output directory for malware CFGs')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--metadata-file', type=str, default='data/extraction_metadata.json',
                        help='Output file for extraction metadata')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CFG Batch Extraction Tool")
    logger.info("=" * 60)

    all_metadata = {
        'benign': [],
        'malware': [],
        'extraction_date': datetime.now().isoformat()
    }

    # Process benign binaries
    if os.path.exists(args.benign_dir):
        logger.info(f"\nProcessing benign binaries from {args.benign_dir}")
        all_metadata['benign'] = process_directory(
            args.benign_dir,
            args.output_benign,
            'benign',
            args.workers
        )
    else:
        logger.warning(f"Benign directory not found: {args.benign_dir}")

    # Process malware binaries
    if os.path.exists(args.malware_dir):
        logger.info(f"\nProcessing malware binaries from {args.malware_dir}")
        all_metadata['malware'] = process_directory(
            args.malware_dir,
            args.output_malware,
            'malware',
            args.workers
        )
    else:
        logger.warning(f"Malware directory not found: {args.malware_dir}")

    # Save metadata
    os.makedirs(os.path.dirname(args.metadata_file), exist_ok=True)
    with open(args.metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"\nMetadata saved to {args.metadata_file}")
    logger.info("=" * 60)
    logger.info("Extraction complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

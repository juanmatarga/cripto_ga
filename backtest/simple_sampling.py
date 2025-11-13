"""
Simple Random Sampling for Pattern Evaluation.

Creates non-overlapping random windows for robustness testing.
Better suited for GA pattern discovery than walk-forward optimization.
"""

import pandas as pd
import numpy as np
from typing import List
import logging
import random

logger = logging.getLogger(__name__)

def create_simple_windows(data: pd.DataFrame,
                         n_windows: int = 5,
                         window_months: int = 1,
                         seed: int = 42) -> List[pd.DataFrame]:
    """
    Create N non-overlapping random windows of M months each.

    Strategy:
        1. Split data into chunks of window_months
        2. Randomly sample n_windows chunks without overlap
        3. Return list of DataFrames

    Args:
        data: Full dataset with DatetimeIndex
        n_windows: Number of windows to create
        window_months: Duration of each window in months
        seed: Random seed for reproducibility

    Returns:
        List of DataFrames, each representing one window

    Example:
        >>> windows = create_simple_windows(data, n_windows=5, window_months=1)
        >>> len(windows)
        5
        >>> len(windows[0])  # ~2880 bars for 1 month in 15min
        2880
    """
    logger.info("Creating simple random windows...")
    logger.info(f"  Target: {n_windows} windows × {window_months} month(s) each")

    random.seed(seed)
    np.random.seed(seed)

    # Add year-month column for grouping
    data = data.copy()
    data['year_month'] = data.index.to_period('M')
    unique_months = data['year_month'].unique()

    logger.info(f"  Available months: {len(unique_months)} ({unique_months[0]} to {unique_months[-1]})")

    # Create all possible chunks
    all_chunks = []
    chunk_labels = []

    for i in range(0, len(unique_months) - window_months + 1):
        # Get consecutive months for this chunk
        chunk_months = unique_months[i:i + window_months]
        chunk_data = data[data['year_month'].isin(chunk_months)].copy()

        # Only keep chunks with sufficient data
        if len(chunk_data) > 500:  # Minimum 500 bars (~3.5 days in 15min)
            all_chunks.append(chunk_data.drop(columns=['year_month']))
            chunk_labels.append(f"{chunk_months[0]}")

    logger.info(f"  Created {len(all_chunks)} potential chunks")

    # Sample non-overlapping chunks
    if len(all_chunks) < n_windows:
        logger.warning(f"  Only {len(all_chunks)} chunks available, using all")
        selected_chunks = all_chunks
        selected_labels = chunk_labels
    else:
        # Random sampling without replacement
        sample_indices = random.sample(range(len(all_chunks)), n_windows)
        selected_chunks = [all_chunks[i] for i in sorted(sample_indices)]
        selected_labels = [chunk_labels[i] for i in sorted(sample_indices)]

    logger.info(f"  Selected {len(selected_chunks)} windows:")
    for i, (chunk, label) in enumerate(zip(selected_chunks, selected_labels)):
        logger.info(f"    Window {i+1}: {label} ({len(chunk)} bars, "
                   f"{chunk.index[0].date()} to {chunk.index[-1].date()})")

    return selected_chunks


def stratified_sampling(data: pd.DataFrame,
                       n_windows: int = 5,
                       window_months: int = 1,
                       seed: int = 42) -> List[pd.DataFrame]:
    """
    Create windows stratified by market regime (volatility).

    Ensures windows cover different market conditions:
        - Low volatility periods
        - Medium volatility periods
        - High volatility periods

    Args:
        data: Full dataset
        n_windows: Number of windows
        window_months: Duration per window
        seed: Random seed

    Returns:
        List of DataFrames stratified by volatility regime
    """
    logger.info("Creating stratified windows by volatility...")

    random.seed(seed)
    np.random.seed(seed)

    # Calculate rolling volatility for stratification
    data = data.copy()
    data['returns'] = data['Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=96).std()  # Daily vol in 15min

    # Add year-month
    data['year_month'] = data.index.to_period('M')
    unique_months = data['year_month'].unique()

    # Create chunks with volatility labels
    chunks_with_vol = []

    for i in range(0, len(unique_months) - window_months + 1):
        chunk_months = unique_months[i:i + window_months]
        chunk_data = data[data['year_month'].isin(chunk_months)].copy()

        if len(chunk_data) > 500:
            # Calculate average volatility for this chunk
            avg_vol = chunk_data['volatility'].mean()
            chunks_with_vol.append({
                'data': chunk_data.drop(columns=['year_month', 'returns', 'volatility']),
                'volatility': avg_vol,
                'label': f"{chunk_months[0]}"
            })

    # Sort by volatility
    chunks_with_vol.sort(key=lambda x: x['volatility'])

    # Divide into volatility terciles
    n_chunks = len(chunks_with_vol)
    low_vol = chunks_with_vol[:n_chunks//3]
    med_vol = chunks_with_vol[n_chunks//3:2*n_chunks//3]
    high_vol = chunks_with_vol[2*n_chunks//3:]

    logger.info(f"  Low vol chunks: {len(low_vol)}")
    logger.info(f"  Med vol chunks: {len(med_vol)}")
    logger.info(f"  High vol chunks: {len(high_vol)}")

    # Sample from each regime
    selected = []
    per_regime = n_windows // 3
    remainder = n_windows % 3

    # Sample from each tercile
    if len(low_vol) >= per_regime:
        selected.extend(random.sample(low_vol, per_regime))
    if len(med_vol) >= per_regime:
        selected.extend(random.sample(med_vol, per_regime))
    if len(high_vol) >= per_regime + remainder:
        selected.extend(random.sample(high_vol, per_regime + remainder))

    # Sort by time
    selected.sort(key=lambda x: x['data'].index[0])

    logger.info(f"  Selected {len(selected)} stratified windows:")
    for i, chunk_info in enumerate(selected):
        chunk = chunk_info['data']
        logger.info(f"    Window {i+1}: {chunk_info['label']} "
                   f"(vol={chunk_info['volatility']:.6f}, {len(chunk)} bars)")

    return [c['data'] for c in selected]

"""
Central configuration for the N-D parallelism simulations.
Ensures consistent work units and data sizes across different strategies
for fair time comparisons.
"""

# --- Computational Work ---
# Total compute units for processing one data item through the entire model (fwd/bwd).
TOTAL_COMPUTE_PER_ITEM = 20.0

# --- Data Sizes (Arbitrary Units) ---
# Simulated size of model gradients for one replica (for DP All-Reduce).
GRADIENT_SIZE_PER_MODEL = 500.0

# --- Batching and Workers ---
# Number of data items to process in total.
NUM_DATA_ITEMS = 16

# Default number of parallel workers (simulated GPUs/ranks) for parallel strategies.
NUM_WORKERS = 4

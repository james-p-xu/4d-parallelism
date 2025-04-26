import multiprocessing
import time

import numpy as np

from gpu import GPU
from simulation_config import (
    GRADIENT_SIZE_PER_MODEL,
    NUM_DATA_ITEMS,
    NUM_WORKERS,
    TOTAL_COMPUTE_PER_ITEM,
)


def simulate_dp_worker(rank, data_chunk):
    """
    Task executed by each data parallel worker process (simulated GPU).
    Processes a chunk of data.
    Args:
        rank (int): The rank of this worker.
        data_chunk (list): The portion of data this worker should process.
    """
    gpu = GPU(rank=rank)
    results_chunk = []

    gpu._log(f"Starting processing {len(data_chunk)} items.")

    for item in data_chunk:
        task_desc = f"Process item {item}"

        gpu.compute(task_desc, TOTAL_COMPUTE_PER_ITEM)

        # The actual result calculation (separate from time simulation)
        result = item * item
        results_chunk.append(result)

    gpu._log(f"Finished processing chunk. Returning {len(results_chunk)} results.")

    return results_chunk


def main():
    num_items = NUM_DATA_ITEMS
    num_workers = NUM_WORKERS
    data = range(1, num_items + 1)
    data_list = list(data)

    print(
        f"Starting Data Parallelism (1D) on {len(data_list)} items using {num_workers} ranks..."
    )
    print(
        f"(Total compute units per item: {TOTAL_COMPUTE_PER_ITEM}, Grad size: {GRADIENT_SIZE_PER_MODEL})"
    )
    start_time = time.time()

    # Prepare iterable of (rank, chunk) tuples
    data_chunks = np.array_split(data_list, num_workers)
    pool_data = list(enumerate(data_chunks))

    with multiprocessing.Pool(processes=num_workers) as pool:
        list_of_results_chunks = pool.starmap(simulate_dp_worker, pool_data)

    results = [item for sublist in list_of_results_chunks for item in sublist]

    computation_end_time = time.time()
    print(
        f"Computation finished. Time taken: {computation_end_time - start_time:.2f} seconds"
    )

    print(
        f"Simulating gradient synchronization (e.g., All-Reduce) across {num_workers} ranks..."
    )
    sync_start_time = time.time()

    # Use dummy GPUs in main process only for timing simulation
    dummy_gpus = [GPU(rank=r) for r in range(num_workers)]

    # Simulate the time cost of ring all-reduce
    for r in range(num_workers):
        dummy_gpus[r].send(
            target_rank=(r + 1) % num_workers,
            data_description=f"Gradients from Rank {r}",
            data_size_units=GRADIENT_SIZE_PER_MODEL,
        )
        dummy_gpus[r].receive(
            source_rank=(r - 1 + num_workers) % num_workers,
            data_description=f"Aggregated Gradients",
            data_size_units=GRADIENT_SIZE_PER_MODEL,
        )

    sync_end_time = time.time()
    print(
        f"Gradient synchronization simulation finished. Time taken: {sync_end_time - sync_start_time:.2f} seconds"
    )

    total_end_time = time.time()
    print("\nResults:", results)
    print(
        f"Total data parallel execution time (incl. simulated sync): {total_end_time - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()

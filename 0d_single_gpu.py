import time

from gpu import GPU
from simulation_config import NUM_DATA_ITEMS, TOTAL_COMPUTE_PER_ITEM


def simulate_work_on_gpu(gpu: GPU, item):
    """Simulates computation for a data item on a specific GPU."""
    compute_units = TOTAL_COMPUTE_PER_ITEM
    task_description = f"Process item {item}"

    gpu.compute(task_description, compute_units)

    # The actual result calculation (separate from time simulation)
    result = item * item

    gpu._log(f"Work on item {item} complete, result: {result}")

    return result


def main():
    gpu0 = GPU(rank=0)
    num_items = NUM_DATA_ITEMS

    print(f"Starting sequential processing (0D - Single GPU) on {num_items} items...")
    start_time = time.time()

    results = [simulate_work_on_gpu(gpu0, item) for item in range(num_items)]

    end_time = time.time()
    print("\nResults:", results)
    print(f"Total sequential execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

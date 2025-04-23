# N-D Parallelism Explained (0D to 4D)

This repository explores **N-dimesional parallelism**, commonly discussed in the context of **training large-scale machine learning models**. We start from a baseline (0D, sequential execution) and build up to 4D parallelism, illustrating each concept with simple Python examples using a **"GPU"** that incorporates compute and network latency.

The goal is to build an intuitive understanding of the various parallelism techniques, including **recent advancements** used to efficiently train today's increasingly large machine learning models, by **distributing computation** and **managing communication** across multiple devices.

## GPU Simulator (`gpu.py`)

To illustrate the time-based trade-offs of different parallelism strategies without requiring actual GPU hardware or complex distributed libraries, this repository uses a simplified simulator defined in [`gpu.py`](./gpu.py).

This file contains a `GPU` class that simulates:

*   **Computation Time:** The `gpu.compute(description, compute_units)` method simulates work by pausing execution for a duration calculated based on `compute_units` and a base `SECONDS_PER_COMPUTE_UNIT` constant
*   **Network Communication Time:**
    *   `gpu.send(target_rank, description, data_size_units)` simulates the time taken to send data, considering both fixed latency (`NETWORK_BASE_LATENCY_SECONDS`) and bandwidth (`SECONDS_PER_DATA_UNIT * data_size_units`)
    *   `gpu.receive(source_rank, description, data_size_units)` simulates the latency associated with receiving data and acknowledgments

This simulator focuses purely on **time delays** for compute and network operations. It does **not** simulate:
*   Actual data transfer (which is handled by `multiprocessing` mechanisms like queues)
*   GPU memory constraints or memory bandwidth
*   Detailed network topology or synchronization primitives

Its purpose is to provide a simple way to model the performance characteristics (computation vs. communication) of the different parallelism dimensions conceptually. Each simulated `GPU` instance is identified by its `global_rank`.

## Dimensions of Parallelism

*(Note: All examples use `gpu.py` to model time delays for computation and network communication)*

### 0D) Sequential / Single GPU Execution

This is a baseline where all computations for a batch happen sequentially on a single simulated device.

*   **Concept:** Process data items or perform model operations serially on one device
*   **Simulation:** [`0d_single_gpu.py`](./0d_single_gpu.py) uses a single `GPU` instance to process items one by one, showing the total compute time without any parallelism

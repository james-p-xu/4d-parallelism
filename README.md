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

### 1D) Data Parallelism (DP)

This is perhaps the most common strategy for scaling training. The core idea is to replicate the entire model on multiple devices (ranks). Each device processes a different slice (mini-batch) of the global data batch concurrently.

*   **Concept:** Replicate model, split data batch, synchronize gradients
*   **Workflow:**
    1.  Each rank computes the forward pass on its local mini-batch
    2.  Each rank computes the backward pass, generating local gradients
    3.  Gradients are synchronized and averaged across all ranks. A common efficient algorithm for this is **Ring All-Reduce**
    4.  Each rank updates its local model copy using the synchronized gradients
*   **Pros:** Easy to implement using PyTorch's DDP. Scales well if computation time significantly outweighs communication time
*   **Cons:** Memory intensive (each rank holds the full model, gradients, and optimizer states). Communication overhead for gradient synchronization can become a bottleneck, especially with many devices or slow networks
*   **Simulation:** [`1d_data_parallel.py`](./1d_data_parallel.py) simulates this using multiple processes (ranks). Each rank processes a data chunk in parallel. A separate step simulates the time cost of gradient synchronization using Ring All-Reduce
*   **Resources:**
    1. Getting Started with Distributed Data Parallel [PyTorch Docs](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
    2. Bringing HPC Techniques to Deep Learning (Baidu's All-Reduce) [blog](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)

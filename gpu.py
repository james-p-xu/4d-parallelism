import os
import time

# --- Simulation Constants ---

# Represents the simulated processing speed of the GPU.
# Time in seconds for the GPU to perform one abstract "unit" of computation.
# Lower value ~ faster GPU computation
SECONDS_PER_COMPUTE_UNIT = 0.1

# Represents the simulated fixed overhead for initiating any network transfer.
# Time in seconds for the network latency component (e.g. round-trip time, handshake).
# Applied to both send and receive.
NETWORK_BASE_LATENCY_SECONDS = 0.05

# Represents the simulated speed of the network connection.
# Time in seconds to transfer one abstract "unit" of data.
# Lower value ~ higher simulated bandwidth
SECONDS_PER_DATA_UNIT = 0.01


class GPU:
    """
    Simulates a single GPU device.
    """

    def __init__(self, rank):
        self.global_rank = rank
        self.pid = os.getpid()
        print(f"[PID {self.pid}] Rank {self.global_rank}: Initialized (Simulated GPU).")

    def _log(self, message):
        """Helper for logging messages with Rank ID and PID."""
        print(f"[PID {self.pid}] Rank {self.global_rank}: {message}")

    def compute(self, task_description: str, compute_units: float):
        """
        Simulates performing some computation.
        Args:
            task_description: A string describing the task.
            compute_units: A float representing the amount of work.
        """
        self._log(f"Starting compute: {task_description} ({compute_units} units)")

        simulated_compute_time = SECONDS_PER_COMPUTE_UNIT * compute_units
        time.sleep(simulated_compute_time)

        self._log(
            f"Finished compute: {task_description} (took {simulated_compute_time:.3f}s)"
        )

        # No data is returned; the focus is time simulation
        return None

    def send(self, target_rank: int, data_description: str, data_size_units: float):
        """
        Simulates sending data to another GPU (identified by rank).
        Args:
            target_rank: The rank of the destination GPU.
            data_description: A string describing the data.
            data_size_units: A float representing data size.
        """
        self._log(
            f"Initiating send: {data_description} ({data_size_units} units) to Rank {target_rank}"
        )

        # Simulate latency + bandwidth delay
        transfer_time = NETWORK_BASE_LATENCY_SECONDS + (
            SECONDS_PER_DATA_UNIT * data_size_units
        )
        time.sleep(transfer_time)

        self._log(
            f"Finished send: {data_description} to Rank {target_rank} (took {transfer_time:.3f}s)"
        )

        # Actual data transfer handled by external queues
        return None

    def receive(self, source_rank: int, data_description: str, data_size_units: float):
        """
        Simulates receiving data from another GPU (identified by rank).
        Note: Primarily simulates the time delay associated with receiving.
        Args:
            source_rank: The rank of the source GPU.
            data_description: A string describing the data (for logging).
            data_size_units: A float representing data size.
        """
        self._log(
            f"Expecting receive: {data_description} ({data_size_units} units) from Rank {source_rank}"
        )

        # Simulate acknowledgment/processing overhead latency
        receive_latency = NETWORK_BASE_LATENCY_SECONDS
        time.sleep(receive_latency)

        self._log(
            f"Finished receive: {data_description} from Rank {source_rank} (ack delay {receive_latency:.3f}s)"
        )

        # Actual data reception handled by external queues
        return None


def test_gpu():
    print("--- GPU Simulator Test ---")

    gpu_rank0 = GPU(rank=0)
    gpu_rank1 = GPU(
        rank=1
    )  # Simulate another GPU (in the same process for this example)

    print("\nSimulating computation on Rank 0:")

    gpu_rank0.compute(task_description="Layer 1 Forward Pass", compute_units=5)

    print("\nSimulating data transfer from Rank 0 to Rank 1:")

    data_to_send = "Activations layer 1"
    data_size = 100
    gpu_rank0.send(
        target_rank=1, data_description=data_to_send, data_size_units=data_size
    )
    gpu_rank1.receive(
        source_rank=0, data_description=data_to_send, data_size_units=data_size
    )

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    test_gpu()

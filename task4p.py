
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import statistics
import os

# ==============================
# Configuration
# ==============================
ARRAY_SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3
DTYPE = np.dtype(np.int64)


def receiver_process(conn):
    """Receive arrays through IPC or attach to shared memory."""
    while True:
        message = conn.recv()

        if message[0] == "STOP":
            break

        mode = message[0]

        if mode == "IPC":
            array = message[1]
            result = (array.size, int(array[-1]))

        elif mode == "SHM":
            _, shm_name, size = message
            shm = shared_memory.SharedMemory(name=shm_name)

            try:
                array = np.ndarray(
                    (size,), dtype=DTYPE, buffer=shm.buf
                )
                result = (array.size, int(array[-1]))
            finally:
                shm.close()

        conn.send(result)

    conn.close()


def benchmark_ipc(conn, array):
    """Measure sending a serialized NumPy array through a pipe."""
    start = time.perf_counter()

    conn.send(("IPC", array))
    received_size, last_value = conn.recv()

    elapsed_ms = (time.perf_counter() - start) * 1000

    assert received_size == array.size
    assert last_value == int(array[-1])

    return elapsed_ms


def benchmark_shared_memory(conn, array):
    """Measure shared-memory allocation, copy, and receiver access."""
    start = time.perf_counter()

    shm = shared_memory.SharedMemory(
        create=True,
        size=array.nbytes
    )

    try:
        shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=shm.buf
        )

        # Copy source data into shared memory
        shared_array[:] = array

        # Only metadata is sent through the pipe
        conn.send(("SHM", shm.name, array.size))
        received_size, last_value = conn.recv()

        assert received_size == array.size
        assert last_value == int(array[-1])

    finally:
        shm.close()
        shm.unlink()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


def main():
    print("TASK 4: IPC Serialization vs Shared Memory")
    print(f"Repeats per size: {REPEATS}")
    print("-" * 78)
    print(
        f"{'N':>12} | {'IPC (ms)':>12} | "
        f"{'Shared Memory (ms)':>19} | {'Faster Path':>14}"
    )
    print("-" * 78)

    output_lines = [
        "TASK 4: IPC Serialization vs Shared Memory",
        f"Repeats per size: {REPEATS}",
        "Times include data transfer and receiver access.",
        "Shared memory includes allocation and copying into shared memory.",
        "",
        "N | IPC (ms) | Shared Memory (ms) | Faster Path"
    ]

    # Start the receiver once so process startup is not timed
    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(
        target=receiver_process,
        args=(child_conn,)
    )
    process.start()
    child_conn.close()

    try:
        for n in ARRAY_SIZES:
            rng = np.random.default_rng(seed=12345 + n)
            array = rng.integers(
                1, 101, size=n, dtype=DTYPE
            )

            ipc_times = []
            shm_times = []

            # Warm up both paths
            benchmark_ipc(parent_conn, array[:100])
            benchmark_shared_memory(parent_conn, array[:100])

            for _ in range(REPEATS):
                ipc_times.append(benchmark_ipc(parent_conn, array))
                shm_times.append(
                    benchmark_shared_memory(parent_conn, array)
                )

            ipc_ms = statistics.median(ipc_times)
            shm_ms = statistics.median(shm_times)

            faster = (
                "IPC" if ipc_ms < shm_ms
                else "Shared Memory" if shm_ms < ipc_ms
                else "Tie"
            )

            line = (
                f"{n:12,d} | {ipc_ms:12.3f} | "
                f"{shm_ms:19.3f} | {faster:>14}"
            )
            print(line)

            output_lines.append(
                f"{n} | {ipc_ms:.3f} | {shm_ms:.3f} | {faster}"
            )

    finally:
        parent_conn.send(("STOP",))
        process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            process.join()

        parent_conn.close()

    output_lines.append("")
    output_lines.append(
        "IPC uses Pickle serialization through a multiprocessing pipe."
    )
    output_lines.append(
        "Shared memory avoids sending the full array through the pipe, "
        "but includes allocation and a source-to-shared-memory copy."
    )

    with open("task4p_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines) + "\n")

    print("-" * 78)
    print("Results saved to task4p_output.txt")


if __name__ == "__main__":
    mp.freeze_support()
    main()

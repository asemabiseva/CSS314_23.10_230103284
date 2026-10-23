
import numpy as np
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# ==============================
# Configuration
# ==============================
N = 2_000_000
STUDENT_ID = 230103284  # Replace with your actual Student ID
THREAD_INDEX = 0
NUM_THREADS = 4

SEED = (STUDENT_ID * 10) + THREAD_INDEX


# ==============================
# Generate input array
# ==============================
rng = np.random.default_rng(SEED)
input_array = rng.integers(1, 11, size=N, dtype=np.int64)


# ==============================
# Serial inclusive prefix scan
# ==============================
start = time.perf_counter()

serial_output = np.cumsum(input_array, dtype=np.int64)

serial_time = (time.perf_counter() - start) * 1000


# ==============================
# Parallel chunked prefix scan
# ==============================
def local_scan(chunk):
    """Calculate the inclusive prefix sum of one chunk."""
    return np.cumsum(chunk, dtype=np.int64)


def apply_offset(args):
    """Add the preceding chunks' total to a chunk."""
    chunk, offset = args

    if offset != 0:
        chunk += offset

    return chunk


start = time.perf_counter()

# Divide the input into approximately equal chunks
chunks = np.array_split(input_array, NUM_THREADS)

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:

    # Pass 1: Compute local prefix sums in parallel
    local_outputs = list(executor.map(local_scan, chunks))

    # Calculate each chunk's total
    chunk_sums = np.array(
        [chunk[-1] for chunk in local_outputs],
        dtype=np.int64
    )

    # Offset reduction: cumulative totals of the chunks
    cumulative_chunk_sums = np.cumsum(chunk_sums)

    # Offset for each chunk is the total of all preceding chunks
    offsets = np.zeros(NUM_THREADS, dtype=np.int64)
    offsets[1:] = cumulative_chunk_sums[:-1]

    # Pass 2: Apply offsets in parallel
    tasks = [
        (local_outputs[i], offsets[i])
        for i in range(NUM_THREADS)
    ]

    adjusted_chunks = list(executor.map(apply_offset, tasks))

# Combine the adjusted chunks into the final output
parallel_output = np.concatenate(adjusted_chunks)

parallel_time = (time.perf_counter() - start) * 1000


# ==============================
# Verification
# ==============================
try:
    assert parallel_output[-1] == serial_output[-1], (
        f"Verification failed: parallel={parallel_output[-1]}, "
        f"serial={serial_output[-1]}"
    )

    # Additional check: verify every prefix sum
    assert np.array_equal(parallel_output, serial_output), (
        "The prefix arrays differ."
    )

    verification_status = "PASS"

except AssertionError as error:
    verification_status = "FAIL"

    output_text = (
        f"Serial Time (ms): {serial_time:.3f}\n"
        f"Parallel Time (ms): {parallel_time:.3f}\n"
        f"Speedup: {serial_time / parallel_time:.4f}x\n"
        f"Verification Status: FAIL\n"
        f"Error: {error}\n"
    )

    print(output_text)

    with open("task3_output.txt", "w", encoding="utf-8") as file:
        file.write(output_text)

    sys.exit(1)


# ==============================
# Results
# ==============================
speedup = serial_time / parallel_time

output_text = (
    f"N: {N}\n"
    f"Seed: {SEED}\n"
    f"Number of threads: {NUM_THREADS}\n"
    f"Serial Time (ms): {serial_time:.3f}\n"
    f"Parallel Time (ms): {parallel_time:.3f}\n"
    f"Speedup: {speedup:.4f}x\n"
    f"Verification Status: {verification_status}\n"
    f"Serial final sum: {serial_output[-1]}\n"
    f"Parallel final sum: {parallel_output[-1]}\n"
)

print(output_text)

with open("task3_output.txt", "w", encoding="utf-8") as file:
    file.write(output_text)

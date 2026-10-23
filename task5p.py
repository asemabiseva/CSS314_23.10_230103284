
import os
import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# Configuration
# ==========================================
NUM_TASKS = 10_000
HASH_ITERATIONS = 500
PAYLOAD = b"A" * 1024  # 1 KB of data
REPEATS = 3

CORES = os.cpu_count() or 1

POOL_SIZES = sorted(set([
    1,
    max(1, CORES // 2),
    CORES,
    CORES * 2,
    CORES * 4,
    CORES * 16,
    256
]))


# ==========================================
# CPU-bound task
# ==========================================
def hashing_task(task_id):
    """Hash a 1 KB payload 500 times."""
    result = PAYLOAD

    for _ in range(HASH_ITERATIONS):
        result = hashlib.sha256(PAYLOAD).digest()

    return result


# ==========================================
# Benchmark one pool size
# ==========================================
def benchmark_pool(pool_size):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            results = list(
                executor.map(hashing_task, range(NUM_TASKS))
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Ensure all tasks completed successfully
        assert len(results) == NUM_TASKS

        times.append(elapsed_ms)

    return statistics.median(times)


# ==========================================
# Main benchmark
# ==========================================
def main():
    print("TASK 5: Thread Oversubscription Threshold")
    print(f"CPU cores reported by OS: {CORES}")
    print(f"Number of tasks: {NUM_TASKS:,}")
    print(f"Hashes per task: {HASH_ITERATIONS}")
    print(f"Runs per pool size: {REPEATS}")
    print()

    print(
        f"{'Pool Size':>10} | "
        f"{'Execution Time (ms)':>19} | "
        f"{'Speedup':>8} | "
        f"{'Status'}"
    )
    print("-" * 76)

    timings = {}

    for pool_size in POOL_SIZES:
        timings[pool_size] = benchmark_pool(pool_size)

    baseline = timings[1]
    best_time_so_far = baseline
    previous_time = None
    threshold_found = False
    output_lines = [
        "TASK 5: Thread Oversubscription Threshold",
        f"CPU cores reported by OS: {CORES}",
        f"Number of tasks: {NUM_TASKS}",
        f"Hashes per task: {HASH_ITERATIONS}",
        f"Runs per pool size: {REPEATS}",
        "",
        "Pool Size | Execution Time (ms) | Speedup | Status"
    ]

    for pool_size in POOL_SIZES:
        elapsed_ms = timings[pool_size]
        speedup = baseline / elapsed_ms

        if pool_size == 1:
            status = "BASELINE"
        elif not threshold_found and elapsed_ms > previous_time:
            status = "WORSENING START"
            threshold_found = True
        elif elapsed_ms < best_time_so_far:
            status = "IMPROVED"
        elif elapsed_ms > previous_time:
            status = "SLOWER"
        else:
            status = "NO CLEAR CHANGE"

        if elapsed_ms < best_time_so_far:
            best_time_so_far = elapsed_ms

        line = (
            f"{pool_size:10d} | "
            f"{elapsed_ms:19.3f} | "
            f"{speedup:7.3f}x | "
            f"{status}"
        )

        print(line)
        output_lines.append(line)

        previous_time = elapsed_ms

    print()

    if threshold_found:
        threshold = next(
            size for i, size in enumerate(POOL_SIZES)
            if i > 0 and timings[size] > timings[POOL_SIZES[i - 1]]
        )
        summary = (
            f"First observed worsening: pool size {threshold} threads."
        )
    else:
        summary = (
            "No worsening point observed between adjacent tested "
            "pool sizes."
        )

    print(summary)
    output_lines.extend(["", summary])
    output_lines.append(
        "Note: timings include thread-pool creation and shutdown. "
        "The first worsening point is an observed benchmark result, "
        "not proof that OS context switching alone caused the slowdown."
    )

    with open("task5p_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines) + "\n")

    print("Results saved to task5p_output.txt")


if __name__ == "__main__":
    main()

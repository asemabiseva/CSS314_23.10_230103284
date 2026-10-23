"""
Task 1: Row-Major vs. Column-Major Parallel Cache Traversal
Run: python task1.py
Dependency: numpy, psutil
"""
import os
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

MATRIX_SIZE = 8_000
WORKER_COUNT = None  # Uses all detected physical CPU cores when psutil is available.


def _sum_rows(args):
    shm_name, shape, start, stop = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        matrix = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
        # Complete consecutive rows: contiguous memory traversal.
        return float(np.sum(matrix[start:stop, :]))
    finally:
        shm.close()


def _sum_columns_strided(args):
    shm_name, shape, start, stop = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        matrix = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
        # Process one column at a time. In a C-order matrix, successive values
        # down a column are separated by MATRIX_SIZE float64 elements.
        subtotal = 0.0
        for col in range(start, stop):
            subtotal += float(np.sum(matrix[:, col]))
        return subtotal
    finally:
        shm.close()


def split_ranges(length, workers):
    """Create balanced, non-empty ranges."""
    workers = max(1, min(workers, length))
    q, r = divmod(length, workers)
    ranges = []
    start = 0
    for i in range(workers):
        stop = start + q + (1 if i < r else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def main():
    student_id = 230103284

    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except ImportError:
        physical_cores = os.cpu_count() or 1
        logical_cores = os.cpu_count() or 1
        print("Note: install psutil for separate physical/logical core detection: pip install psutil")

    workers = max(1, min(physical_cores, MATRIX_SIZE))
    shape = (MATRIX_SIZE, MATRIX_SIZE)
    nbytes = MATRIX_SIZE * MATRIX_SIZE * np.dtype(np.float64).itemsize

    print(f"\nStudent ID: {student_id}")
    print(f"Logical core count: {logical_cores}")
    print(f"Physical cores / worker processes: {physical_cores} / {workers}")
    print(f"Matrix: {MATRIX_SIZE} x {MATRIX_SIZE}, float64 ({nbytes / (1024**2):.1f} MiB)")
    print("Generating matrix...")

    seed = student_id
    rng = np.random.default_rng(seed)
    matrix = rng.random(shape, dtype=np.float64)

    # Shared memory avoids copying the 512 MB matrix into each worker process.
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        shared_matrix = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
        shared_matrix[:] = matrix
        del matrix

        row_ranges = split_ranges(MATRIX_SIZE, workers)
        col_ranges = split_ranges(MATRIX_SIZE, workers)
        row_tasks = [(shm.name, shape, a, b) for a, b in row_ranges]
        col_tasks = [(shm.name, shape, a, b) for a, b in col_ranges]

        # Warm-up is intentionally omitted so each reported time measures the
        # requested full parallel sum, including process-pool startup.
        ctx = mp.get_context("spawn")

        start = time.perf_counter()
        with ctx.Pool(processes=workers) as pool:
            row_parts = pool.map(_sum_rows, row_tasks)
        row_time_ms = (time.perf_counter() - start) * 1000.0
        row_sum = sum(row_parts)

        start = time.perf_counter()
        with ctx.Pool(processes=workers) as pool:
            col_parts = pool.map(_sum_columns_strided, col_tasks)
        col_time_ms = (time.perf_counter() - start) * 1000.0
        col_sum = sum(col_parts)

        penalty = col_time_ms / row_time_ms if row_time_ms else float("inf")
        sums_match = np.isclose(row_sum, col_sum, rtol=1e-10, atol=1e-6)

        lines = [
            "Row-Major vs. Column-Major Parallel Cache Traversal",
            f"Student ID: {student_id}",
            f"Logical core count: {logical_cores}",
            f"Physical cores / worker processes: {physical_cores} / {workers}",
            f"Matrix: {MATRIX_SIZE} x {MATRIX_SIZE} float64",
            f"Mode A (Row-Major) runtime: {row_time_ms:.3f} ms",
            f"Mode B (Column-Major / Strided) runtime: {col_time_ms:.3f} ms",
            f"Cache penalty factor (Mode B / Mode A): {penalty:.4f}x",
            f"Row-major sum: {row_sum:.12f}",
            f"Column-major sum: {col_sum:.12f}",
            f"Sum verification (close): {'PASS' if sums_match else 'CHECK'}",
        ]
        output = "\n".join(lines) + "\n"
        print("\n" + output)
        with open("task1_output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        print("Saved benchmark results to task1_output.txt")
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()

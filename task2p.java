import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;


public class task2p {
    private static final long STUDENT_ID = 230103284L;
    private static final long TOTAL_POINTS = 40_000_000L;
    private static final int THREADS = 4;
    private static final double MATH_PI = Math.PI;

    private static final class SharedRandom {
        private final SplittableRandom random = new SplittableRandom(STUDENT_ID * 10L);
        private final ReentrantLock lock = new ReentrantLock();

        // Both random coordinates are drawn under one lock acquisition, so
        // each point is generated from a consistent serialized PRNG sequence.
        double nextDouble() {
            lock.lock();
            try {
                return random.nextDouble();
            } finally {
                lock.unlock();
            }
        }
    }

    private static long[] chunkSizes(long total, int threads) {
        long[] chunks = new long[threads];
        long base = total / threads;
        long remainder = total % threads;
        for (int i = 0; i < threads; i++) {
            chunks[i] = base + (i < remainder ? 1 : 0);
        }
        return chunks;
    }

    private static long runShared() throws InterruptedException {
        SharedRandom rng = new SharedRandom();
        AtomicLong hits = new AtomicLong(0L);
        long[] chunks = chunkSizes(TOTAL_POINTS, THREADS);
        CountDownLatch done = new CountDownLatch(THREADS);

        for (int t = 0; t < THREADS; t++) {
            final long count = chunks[t];
            Thread worker = new Thread(() -> {
                long localHits = 0L;
                for (long i = 0; i < count; i++) {
                    double x = rng.nextDouble() * 2.0 - 1.0;
                    double y = rng.nextDouble() * 2.0 - 1.0;
                    if (x * x + y * y <= 1.0) {
                        localHits++;
                    }
                }
                hits.addAndGet(localHits);
                done.countDown();
            }, "shared-prng-" + t);
            worker.start();
        }

        done.await();
        return hits.get();
    }

    private static long runIndependent() throws InterruptedException {
        AtomicLong hits = new AtomicLong(0L);
        long[] chunks = chunkSizes(TOTAL_POINTS, THREADS);
        CountDownLatch done = new CountDownLatch(THREADS);

        for (int t = 0; t < THREADS; t++) {
            final int threadIndex = t;
            final long count = chunks[t];
            final long seed = (STUDENT_ID * 10L) + threadIndex;

            Thread worker = new Thread(() -> {
                SplittableRandom rng = new SplittableRandom(seed);
                long localHits = 0L;
                for (long i = 0; i < count; i++) {
                    double x = rng.nextDouble() * 2.0 - 1.0;
                    double y = rng.nextDouble() * 2.0 - 1.0;
                    if (x * x + y * y <= 1.0) {
                        localHits++;
                    }
                }
                hits.addAndGet(localHits);
                done.countDown();
            }, "independent-prng-" + t);
            worker.start();
        }

        done.await();
        return hits.get();
    }

    private static double estimatePi(long hits) {
        return 4.0 * ((double) hits / (double) TOTAL_POINTS);
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("Task 2: Pseudo-Random Generator State Bottleneck");
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("Sample points: " + TOTAL_POINTS);
        System.out.println("Threads: " + THREADS);
        System.out.println();

        // Shared PRNG guarded by a mutex.
        long start = System.nanoTime();
        long sharedHits = runShared();
        double sharedMs = (System.nanoTime() - start) / 1_000_000.0;
        double sharedPi = estimatePi(sharedHits);
        double sharedError = Math.abs(sharedPi - MATH_PI);

        // Each thread owns its own PRNG and uses the required seed formula.
        start = System.nanoTime();
        long independentHits = runIndependent();
        double independentMs = (System.nanoTime() - start) / 1_000_000.0;
        double independentPi = estimatePi(independentHits);
        double independentError = Math.abs(independentPi - MATH_PI);

        double speedup = independentMs > 0.0 ? sharedMs / independentMs : Double.POSITIVE_INFINITY;

        System.out.printf("Shared PRNG:      Computed pi = %.10f, absolute error = %.10f, wall time = %.3f ms%n",
                sharedPi, sharedError, sharedMs);
        System.out.printf("Independent PRNG: Computed pi = %.10f, absolute error = %.10f, wall time = %.3f ms%n",
                independentPi, independentError, independentMs);
        System.out.printf("Speedup (shared time / independent time): %.4fx%n", speedup);

        String output = String.format(
                "Task 2: Pseudo-Random Generator State Bottleneck%n" +
                "Student ID: %d%nSample points: %d%nThreads: %d%n" +
                "Shared PRNG: Computed pi = %.10f, absolute error = %.10f, wall time = %.3f ms%n" +
                "Independent PRNG: Computed pi = %.10f, absolute error = %.10f, wall time = %.3f ms%n" +
                "Speedup (shared time / independent time): %.4fx%n",
                STUDENT_ID, TOTAL_POINTS, THREADS,
                sharedPi, sharedError, sharedMs,
                independentPi, independentError, independentMs, speedup
        );

        try (java.io.PrintWriter writer = new java.io.PrintWriter("task2_output.txt", "UTF-8")) {
            writer.print(output);
        } catch (java.io.IOException e) {
            System.err.println("Could not write task2_output.txt: " + e.getMessage());
            System.exit(1);
        }
        System.out.println("Saved results to task2_output.txt");
    }
}

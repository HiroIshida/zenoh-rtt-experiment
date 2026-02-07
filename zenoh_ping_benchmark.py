#!/usr/bin/env python3
"""
Zenoh Python RTT Benchmark Script - Publisher Side

This script measures the Round-Trip Time (RTT) of zenoh-python using a ping-pong mechanism
with a NumPy array of 30 float64 elements as payload.

This is the PUBLISHER side that sends ping messages and measures RTT.
Run zenoh_pong_responder.py in a separate process for true IPC testing.

Usage: uv run zenoh_ping_benchmark.py
"""

import time
import threading
import numpy as np
import zenoh
from typing import List
import statistics


class ZenohPingBenchmark:
    """RTT benchmark publisher for Zenoh with NumPy array payload."""

    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        self.rtt_times: List[float] = []
        self.session = None
        self.publisher = None
        self.subscriber = None

        # Synchronization primitives
        self.response_received = threading.Event()

        # Ping-pong topics
        self.ping_topic = "benchmark/ping"
        self.pong_topic = "benchmark/pong"

        # Test payload - 30 float64 elements
        self.test_array = np.random.rand(30).astype(np.float64)

    def setup_zenoh(self):
        """Initialize Zenoh session and create publisher/subscriber."""
        # Create Zenoh session
        conf = zenoh.Config()
        self.session = zenoh.open(conf)

        # Create publisher for ping messages
        self.publisher = self.session.declare_publisher(self.ping_topic)

        # Create subscriber for pong responses
        self.subscriber = self.session.declare_subscriber(
            self.pong_topic,
            self.on_pong_received
        )

        # Small delay to ensure subscriber is ready
        time.sleep(0.1)

    def on_pong_received(self, sample):
        """Handler for pong responses."""
        try:
            # Deserialize the received data back to NumPy array
            received_array = np.frombuffer(bytes(sample.payload), dtype=np.float64)

            # Verify the array is correct (basic integrity check)
            if len(received_array) == 30 and np.allclose(received_array, self.test_array):
                # Signal that response was received
                self.response_received.set()
        except Exception as e:
            print(f"Error processing pong: {e}")

    def run_benchmark(self):
        """Execute the RTT benchmark."""
        print("Starting Zenoh RTT Benchmark - Publisher Side")
        print("NOTE: Make sure to run zenoh_pong_responder.py in a separate process!")
        print(f"Payload: NumPy array of {len(self.test_array)} float64 elements ({self.test_array.nbytes} bytes)")
        print(f"Iterations: {self.iterations}")
        print("=" * 60)

        # Setup Zenoh
        self.setup_zenoh()

        # Allow time for everything to initialize
        time.sleep(1.0)

        # Convert test array to bytes for efficient transmission
        payload_bytes = self.test_array.tobytes()

        print("Running benchmark...")
        print("Waiting for responder to be ready...")
        time.sleep(2.0)  # Give responder time to start

        # Run benchmark iterations
        successful_count = 0
        for i in range(self.iterations):
            # Clear the response received event
            self.response_received.clear()

            # Record start time
            start_time = time.perf_counter()

            # Send ping message
            self.publisher.put(payload_bytes)

            # Wait for pong response with timeout
            if self.response_received.wait(timeout=10.0):
                # Record end time
                end_time = time.perf_counter()

                # Calculate RTT in microseconds
                rtt_microseconds = (end_time - start_time) * 1_000_000
                self.rtt_times.append(rtt_microseconds)
                successful_count += 1

                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"Completed {i + 1}/{self.iterations} iterations")
            else:
                print(f"Timeout on iteration {i + 1} - is the responder running?")
                if i < 10:  # Only continue if we're in the first few iterations
                    continue
                else:
                    print("Too many timeouts, stopping benchmark")
                    break

        # Cleanup
        self.subscriber.undeclare()
        self.publisher.undeclare()
        self.session.close()

        # Calculate and display results
        self.display_results(successful_count)

    def display_results(self, successful_count):
        """Calculate and display benchmark results."""
        if not self.rtt_times:
            print("No successful RTT measurements recorded!")
            print("Make sure zenoh_pong_responder.py is running in a separate process.")
            return

        min_rtt = min(self.rtt_times)
        max_rtt = max(self.rtt_times)
        avg_rtt = statistics.mean(self.rtt_times)
        median_rtt = statistics.median(self.rtt_times)

        print("\n" + "=" * 60)
        print("ZENOH IPC BENCHMARK RESULTS (Publisher Side)")
        print("=" * 60)
        print(f"Successful iterations: {successful_count}/{self.iterations}")
        print(f"Minimum RTT: {min_rtt:.2f} μs")
        print(f"Maximum RTT: {max_rtt:.2f} μs")
        print(f"Average RTT: {avg_rtt:.2f} μs")
        print(f"Median RTT: {median_rtt:.2f} μs")

        if len(self.rtt_times) > 1:
            std_dev = statistics.stdev(self.rtt_times)
            print(f"Standard Deviation: {std_dev:.2f} μs")

        print("=" * 60)


def main():
    """Main entry point for the benchmark."""
    try:
        benchmark = ZenohPingBenchmark(iterations=1000)
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
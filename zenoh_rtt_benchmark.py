#!/usr/bin/env python3
"""
Zenoh Python RTT Benchmark Script

This script measures the Round-Trip Time (RTT) of zenoh-python using a ping-pong mechanism
with a NumPy array of 30 float64 elements as payload.

Usage: uv run zenoh_rtt_benchmark.py
"""

import time
import threading
import numpy as np
import zenoh
from typing import List
import statistics


class ZenohRTTBenchmark:
    """RTT benchmark for Zenoh with NumPy array payload."""

    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        self.rtt_times: List[float] = []
        self.session = None
        self.publisher = None
        self.subscriber = None

        # Synchronization primitives
        self.response_received = threading.Event()
        self.benchmark_complete = threading.Event()

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

    def setup_echo_service(self):
        """Set up echo service that responds to ping messages."""
        def echo_handler(sample):
            try:
                # Echo back the same data on pong topic
                self.session.put(self.pong_topic, sample.payload)
            except Exception as e:
                print(f"Error in echo handler: {e}")

        # Subscribe to ping topic for echo responses
        echo_subscriber = self.session.declare_subscriber(self.ping_topic, echo_handler)
        return echo_subscriber

    def run_benchmark(self):
        """Execute the RTT benchmark."""
        print("Starting Zenoh RTT Benchmark")
        print(f"Payload: NumPy array of {len(self.test_array)} float64 elements ({self.test_array.nbytes} bytes)")
        print(f"Iterations: {self.iterations}")
        print("=" * 50)

        # Setup Zenoh
        self.setup_zenoh()

        # Setup echo service
        echo_sub = self.setup_echo_service()

        # Allow time for everything to initialize
        time.sleep(0.5)

        # Convert test array to bytes for efficient transmission
        payload_bytes = self.test_array.tobytes()

        print("Running benchmark...")

        # Run benchmark iterations
        for i in range(self.iterations):
            # Clear the response received event
            self.response_received.clear()

            # Record start time
            start_time = time.perf_counter()

            # Send ping message
            self.publisher.put(payload_bytes)

            # Wait for pong response with timeout
            if self.response_received.wait(timeout=5.0):
                # Record end time
                end_time = time.perf_counter()

                # Calculate RTT in microseconds
                rtt_microseconds = (end_time - start_time) * 1_000_000
                self.rtt_times.append(rtt_microseconds)

                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"Completed {i + 1}/{self.iterations} iterations")
            else:
                print(f"Timeout on iteration {i + 1}")
                break

        # Cleanup
        echo_sub.undeclare()
        self.subscriber.undeclare()
        self.publisher.undeclare()
        self.session.close()

        # Calculate and display results
        self.display_results()

    def display_results(self):
        """Calculate and display benchmark results."""
        if not self.rtt_times:
            print("No successful RTT measurements recorded!")
            return

        successful_iterations = len(self.rtt_times)
        min_rtt = min(self.rtt_times)
        max_rtt = max(self.rtt_times)
        avg_rtt = statistics.mean(self.rtt_times)
        median_rtt = statistics.median(self.rtt_times)

        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Successful iterations: {successful_iterations}/{self.iterations}")
        print(f"Minimum RTT: {min_rtt:.2f} μs")
        print(f"Maximum RTT: {max_rtt:.2f} μs")
        print(f"Average RTT: {avg_rtt:.2f} μs")
        print(f"Median RTT: {median_rtt:.2f} μs")

        if successful_iterations > 1:
            std_dev = statistics.stdev(self.rtt_times)
            print(f"Standard Deviation: {std_dev:.2f} μs")

        print("=" * 50)


def main():
    """Main entry point for the benchmark."""
    try:
        benchmark = ZenohRTTBenchmark(iterations=1000)
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
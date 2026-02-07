#!/usr/bin/env python3
"""
ROS 2 rclpy RTT Benchmark Script (Fair Version)

This script measures the Round-Trip Time (RTT) of ROS 2 rclpy using a ping-pong mechanism
with a NumPy array of 30 float64 elements as payload, using standard Float64MultiArray.

Usage: uv run rclpy_rtt_benchmark.py
"""

import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from typing import List
import statistics

class RclpyRTTBenchmark(Node):
    """RTT benchmark for ROS 2 rclpy with NumPy array payload using Float64MultiArray."""

    def __init__(self, iterations: int = 1000):
        super().__init__('rclpy_rtt_benchmark')
        self.iterations = iterations
        self.rtt_times: List[float] = []
        self.encoding_times: List[float] = []
        self.decoding_times: List[float] = []

        # Synchronization primitives
        self.response_received = threading.Event()
        self.benchmark_complete = threading.Event()

        # Ping-pong topics
        self.ping_topic = "benchmark/ping"
        self.pong_topic = "benchmark/pong"

        # Test payload - 30 float64 elements (same as zenoh benchmark)
        self.test_array = np.random.rand(30).astype(np.float64)

        # ROS 2 publishers and subscribers
        self.ping_publisher = None
        self.pong_subscriber = None
        self.echo_subscriber = None
        self.echo_publisher = None

        # Current iteration tracking
        self.current_iteration = 0

    def setup_ros2(self):
        """Initialize ROS 2 publishers and subscribers."""
        # Create publisher for ping messages
        self.ping_publisher = self.create_publisher(
            Float64MultiArray,
            self.ping_topic,
            10
        )

        # Create subscriber for pong responses
        self.pong_subscriber = self.create_subscription(
            Float64MultiArray,
            self.pong_topic,
            self.on_pong_received,
            10
        )

        # Create echo service (subscriber + publisher)
        self.echo_subscriber = self.create_subscription(
            Float64MultiArray,
            self.ping_topic,
            self.echo_handler,
            10
        )

        self.echo_publisher = self.create_publisher(
            Float64MultiArray,
            self.pong_topic,
            10
        )

        # Small delay to ensure everything is ready
        time.sleep(0.5)

    def numpy_to_msg(self, array: np.ndarray) -> Float64MultiArray:
        """Convert NumPy array to ROS 2 Float64MultiArray message."""
        encode_start = time.perf_counter()

        msg = Float64MultiArray()
        msg.data = array.tolist()

        encode_end = time.perf_counter()
        self.encoding_times.append((encode_end - encode_start) * 1_000_000)  # microseconds

        return msg

    def msg_to_numpy(self, msg: Float64MultiArray) -> np.ndarray:
        """Convert ROS 2 Float64MultiArray message to NumPy array."""
        decode_start = time.perf_counter()

        array = np.array(msg.data, dtype=np.float64)

        decode_end = time.perf_counter()
        self.decoding_times.append((decode_end - decode_start) * 1_000_000)  # microseconds

        return array

    def on_pong_received(self, msg: Float64MultiArray):
        """Handler for pong responses."""
        try:
            # Deserialize the received data back to NumPy array
            received_array = self.msg_to_numpy(msg)

            # Verify the array is correct (basic integrity check)
            if len(received_array) == 30 and np.allclose(received_array, self.test_array):
                # Signal that response was received
                self.response_received.set()
        except Exception as e:
            self.get_logger().error(f"Error processing pong: {e}")

    def echo_handler(self, msg: Float64MultiArray):
        """Echo handler that responds to ping messages."""
        try:
            # Echo back the same data on pong topic
            self.echo_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error in echo handler: {e}")

    def run_benchmark(self):
        """Execute the RTT benchmark."""
        print("Starting ROS 2 rclpy RTT Benchmark (Fair Version)")
        print(f"Payload: NumPy array of {len(self.test_array)} float64 elements ({self.test_array.nbytes} bytes)")
        print(f"Encoding: Standard Float64MultiArray (No Base64 overhead)")
        print(f"Iterations: {self.iterations}")
        print("=" * 50)

        # Setup ROS 2
        self.setup_ros2()

        # Allow time for everything to initialize
        time.sleep(1.0)

        # Convert test array to message
        test_msg = self.numpy_to_msg(self.test_array)

        print("Running benchmark...")

        # Run benchmark iterations
        for i in range(self.iterations):
            # Clear the response received event
            self.response_received.clear()

            # Record start time
            start_time = time.perf_counter()

            # Send ping message
            self.ping_publisher.publish(test_msg)

            # Spin to process callbacks while waiting for response
            timeout_start = time.time()
            timeout_duration = 5.0

            while not self.response_received.is_set():
                # Process ROS 2 callbacks
                # Note: spin_once adds some overhead compared to multi-threaded executor,
                # but kept here to match original structure.
                rclpy.spin_once(self, timeout_sec=0.01)

                # Check timeout
                if time.time() - timeout_start > timeout_duration:
                    print(f"Timeout on iteration {i + 1}")
                    break
            else:
                # Response received within timeout
                end_time = time.perf_counter()

                # Calculate RTT in microseconds
                rtt_microseconds = (end_time - start_time) * 1_000_000
                self.rtt_times.append(rtt_microseconds)

                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"Completed {i + 1}/{self.iterations} iterations")

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

        # Display encoding/decoding statistics
        if self.encoding_times:
            print("\nDATA CONVERSION STATISTICS (List <-> NumPy)")
            print("=" * 50)

            # Encoding statistics
            min_encode = min(self.encoding_times)
            max_encode = max(self.encoding_times)
            avg_encode = statistics.mean(self.encoding_times)
            median_encode = statistics.median(self.encoding_times)

            print("Conversion Time (NumPy array -> Float64MultiArray):")
            print(f"  Minimum: {min_encode:.2f} μs")
            print(f"  Maximum: {max_encode:.2f} μs")
            print(f"  Average: {avg_encode:.2f} μs")
            print(f"  Median: {median_encode:.2f} μs")

            if len(self.encoding_times) > 1:
                std_dev_encode = statistics.stdev(self.encoding_times)
                print(f"  Standard Deviation: {std_dev_encode:.2f} μs")

            # Decoding statistics
            if self.decoding_times:
                min_decode = min(self.decoding_times)
                max_decode = max(self.decoding_times)
                avg_decode = statistics.mean(self.decoding_times)
                median_decode = statistics.median(self.decoding_times)

                print("\nConversion Time (Float64MultiArray -> NumPy array):")
                print(f"  Minimum: {min_decode:.2f} μs")
                print(f"  Maximum: {max_decode:.2f} μs")
                print(f"  Average: {avg_decode:.2f} μs")
                print(f"  Median: {median_decode:.2f} μs")

                if len(self.decoding_times) > 1:
                    std_dev_decode = statistics.stdev(self.decoding_times)
                    print(f"  Standard Deviation: {std_dev_decode:.2f} μs")

                # Total encoding + decoding overhead
                total_encode_decode = avg_encode + avg_decode
                print(f"\nTotal Average Conversion Overhead: {total_encode_decode:.2f} μs")

            print("=" * 50)


def main():
    """Main entry point for the benchmark."""
    rclpy.init()

    try:
        benchmark = RclpyRTTBenchmark(iterations=1000)
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        raise
    finally:
        # Cleanup ROS 2
        try:
            benchmark.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()

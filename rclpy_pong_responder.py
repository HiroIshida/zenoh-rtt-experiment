#!/usr/bin/env python3
"""
ROS 2 rclpy RTT Benchmark Script - Responder Side

This script acts as the responder in the ROS 2 RTT benchmark, echoing back
any ping messages it receives as pong responses.

This is the RESPONDER side that receives ping messages and sends pong responses.
Run rclpy_ping_benchmark.py in a separate process for true IPC testing.

Usage: uv run rclpy_pong_responder.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import signal
import sys


class RclpyPongResponder(Node):
    """RTT benchmark responder for ROS 2 rclpy."""

    def __init__(self):
        super().__init__('rclpy_pong_responder')
        self.running = True
        self.message_count = 0

        # Ping-pong topics
        self.ping_topic = "benchmark/ping"
        self.pong_topic = "benchmark/pong"

        # ROS 2 publishers and subscribers for echo service
        self.echo_subscriber = None
        self.echo_publisher = None

        # Set up signal handling for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.get_logger().info(f"Received signal {signum}, shutting down...")
        self.running = False

    def setup_ros2(self):
        """Initialize ROS 2 publishers and subscribers for echo service."""
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

        self.get_logger().info("ROS 2 responder ready - listening for ping messages...")

    def echo_handler(self, msg: Float64MultiArray):
        """Echo handler that responds to ping messages."""
        try:
            # Echo back the same data on pong topic
            self.echo_publisher.publish(msg)
            self.message_count += 1

            # Progress indicator every 100 messages
            if self.message_count % 100 == 0:
                self.get_logger().info(f"Processed {self.message_count} ping messages")

        except Exception as e:
            self.get_logger().error(f"Error in echo handler: {e}")

    def run_responder(self):
        """Run the responder service."""
        print("Starting ROS 2 rclpy RTT Benchmark - Responder Side")
        print("NOTE: Run rclpy_ping_benchmark.py in a separate process to start the benchmark!")
        print("=" * 80)

        # Setup ROS 2
        self.setup_ros2()

        # Keep running until interrupted
        try:
            while self.running and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=1.0)

        except KeyboardInterrupt:
            self.get_logger().info("Responder interrupted by user")

        finally:
            self.get_logger().info(f"Responder stopped. Processed {self.message_count} total ping messages.")


def main():
    """Main entry point for the responder."""
    rclpy.init()

    try:
        responder = RclpyPongResponder()
        responder.run_responder()
    except Exception as e:
        print(f"Responder failed with error: {e}")
        raise
    finally:
        # Cleanup ROS 2
        try:
            responder.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Zenoh Python RTT Benchmark Script - Responder Side

This script acts as the responder in the Zenoh RTT benchmark, echoing back
any ping messages it receives as pong responses.

This is the RESPONDER side that receives ping messages and sends pong responses.
Run zenoh_ping_benchmark.py in a separate process for true IPC testing.

Usage: uv run zenoh_pong_responder.py
"""

import time
import zenoh
import signal
import sys


class ZenohPongResponder:
    """RTT benchmark responder for Zenoh."""

    def __init__(self):
        self.session = None
        self.echo_subscriber = None
        self.running = True
        self.message_count = 0

        # Ping-pong topics
        self.ping_topic = "benchmark/ping"
        self.pong_topic = "benchmark/pong"

        # Set up signal handling for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False

    def setup_zenoh(self):
        """Initialize Zenoh session and create subscriber."""
        # Create Zenoh session
        conf = zenoh.Config()
        self.session = zenoh.open(conf)

        # Create subscriber for ping messages
        self.echo_subscriber = self.session.declare_subscriber(
            self.ping_topic,
            self.echo_handler
        )

        print("Zenoh responder ready - listening for ping messages...")

    def echo_handler(self, sample):
        """Echo handler that responds to ping messages."""
        try:
            # Echo back the same data on pong topic
            self.session.put(self.pong_topic, sample.payload)
            self.message_count += 1

            # Progress indicator every 100 messages
            if self.message_count % 100 == 0:
                print(f"Processed {self.message_count} ping messages")

        except Exception as e:
            print(f"Error in echo handler: {e}")

    def run_responder(self):
        """Run the responder service."""
        print("Starting Zenoh RTT Benchmark - Responder Side")
        print("NOTE: Run zenoh_ping_benchmark.py in a separate process to start the benchmark!")
        print("=" * 70)

        # Setup Zenoh
        self.setup_zenoh()

        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nResponder interrupted by user")

        finally:
            # Cleanup
            if self.echo_subscriber:
                self.echo_subscriber.undeclare()
            if self.session:
                self.session.close()

            print(f"\nResponder stopped. Processed {self.message_count} total ping messages.")


def main():
    """Main entry point for the responder."""
    try:
        responder = ZenohPongResponder()
        responder.run_responder()
    except Exception as e:
        print(f"Responder failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
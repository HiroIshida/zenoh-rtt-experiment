## Bench round-trip time of zenoh vs rclpy
### zenoh
```
============================================================
ZENOH IPC BENCHMARK RESULTS (Publisher Side)
============================================================
Successful iterations: 1000/1000
Minimum RTT: 78.03 μs
Maximum RTT: 525.62 μs
Average RTT: 117.19 μs
Median RTT: 112.83 μs
Standard Deviation: 26.91 μs
============================================================
```

### rclpy
```
======================================================================
ROS 2 IPC BENCHMARK RESULTS (Publisher Side)
======================================================================
Successful iterations: 1000/1000
Minimum RTT: 246.54 μs
Maximum RTT: 981.04 μs
Average RTT: 315.44 μs
Median RTT: 296.77 μs
Standard Deviation: 63.75 μs
======================================================================

DATA CONVERSION STATISTICS (List <-> NumPy)
======================================================================
Conversion Time (NumPy array -> Float64MultiArray):
  Minimum: 121.06 μs
  Maximum: 121.06 μs
  Average: 121.06 μs
  Median: 121.06 μs

Conversion Time (Float64MultiArray -> NumPy array):
  Minimum: 3.26 μs
  Maximum: 20.36 μs
  Average: 4.21 μs
  Median: 4.01 μs
  Standard Deviation: 0.95 μs

Total Average Conversion Overhead: 125.27 μs
```

## Bench round-trip time of zenoh vs rclpy
### zenoh
```
BENCHMARK RESULTS
==================================================
Successful iterations: 1000/1000
Minimum RTT: 28.54 μs
Maximum RTT: 675.89 μs
Average RTT: 36.89 μs
Median RTT: 32.91 μs
Standard Deviation: 22.43 μs
==================================================
```

### rclpy
```
BENCHMARK RESULTS
==================================================
Successful iterations: 1000/1000
Minimum RTT: 352.29 μs
Maximum RTT: 1547.30 μs
Average RTT: 400.87 μs
Median RTT: 381.78 μs
Standard Deviation: 60.39 μs
==================================================

DATA CONVERSION STATISTICS (List <-> NumPy)
==================================================
Conversion Time (NumPy array -> Float64MultiArray):
  Minimum: 112.50 μs
  Maximum: 112.50 μs
  Average: 112.50 μs
  Median: 112.50 μs

Conversion Time (Float64MultiArray -> NumPy array):
  Minimum: 3.34 μs
  Maximum: 23.28 μs
  Average: 4.04 μs
  Median: 3.78 μs
  Standard Deviation: 1.12 μs

Total Average Conversion Overhead: 116.54 μs
```

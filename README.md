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
Minimum RTT: 294.32 μs
Maximum RTT: 9650.22 μs
Average RTT: 366.58 μs
Median RTT: 338.93 μs
Standard Deviation: 298.81 μs
==================================================

BASE64 ENCODING/DECODING STATISTICS
==================================================
Encoding Time (NumPy array -> base64 string):
  Minimum: 19.19 μs
  Maximum: 19.19 μs
  Average: 19.19 μs
  Median: 19.19 μs

Decoding Time (base64 string -> NumPy array):
  Minimum: 3.43 μs
  Maximum: 18.41 μs
  Average: 4.34 μs
  Median: 4.02 μs
  Standard Deviation: 1.07 μs

Total Average Encoding+Decoding Overhead: 23.53 μs
==================================================
```

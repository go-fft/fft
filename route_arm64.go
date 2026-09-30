package fft

import "math"

// pow2StockhamMaxDefault routes every power of two to the Stockham engine on
// arm64, where it measured faster than the iterative pow2 kernel at every size
// (M4 Max: 256 1.07×, 1024 1.30×, 4096 1.10×, 65536 1.34×).
func pow2StockhamMaxDefault() int { return math.MaxInt }

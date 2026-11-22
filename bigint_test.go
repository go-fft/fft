package fft

import "math"

// Absurd sizes for the overflow and validation tests. Written as 1<<32,
// 1<<40 or 1<<62 they do not compile where int is 32 bits (the 386 lane), so
// each is the smaller of that power and a quarter of the int range: on 64-bit
// targets it is exactly the power, on 32-bit ones 2^30, whose products and
// four-fold multiples still overflow an int, which is what the tests need.
const (
	big32 = min(1<<32, math.MaxInt/2+1)
	big40 = min(1<<40, math.MaxInt/2+1)
	big62 = min(1<<62, math.MaxInt/2+1)
)

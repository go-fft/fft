package fft

import "math"

// pow2StockhamMaxDefault routes every power of two to the Stockham engine on
// arm64, where it measured faster than the iterative pow2 kernel at every size
// (M4 Max: 256 1.07×, 1024 1.30×, 4096 1.10×, 65536 1.34×).
func pow2StockhamMaxDefault() int { return math.MaxInt }

// r8MaxPow2Default and pow2OneRadix8Max factor a power of two 2^e on arm64 as radix-4
// passes, opening with one radix-8 pass when e is odd (never a lone radix-2
// pass). Three rules were timed at every 2^e from 32 to 8192 (2026-09-30):
// A, radix 8 as far as it goes (the rule amd64 keeps); B, radix 4 with a
// radix-2 pass for an odd e; C, this one. Time over the best of the three,
// geometric mean over both CPUs and all sizes: A 1.121 (worst 1.51), B 1.089
// (worst 1.26), C 1.022 (worst 1.11) — Apple M4 and Neoverse-N1 agree. Lengths
// with an odd factor keep rule A, which is neutral or better there.
//
// Above 8192, where those timings stop, rule B is kept: re-timed from 2^13 to
// 2^20, C lost on the M4 at 2^15 (1.19–1.25× B's time, also seen as a 0.87×
// RFFT 65536 end to end), while on Neoverse-N1 it won by only 3–11%. (The even
// exponents, which factor identically under B and C, put the M4's noise at
// ±10% during that run.)
func r8MaxPow2Default() int { return 0 }

const pow2OneRadix8Max = 8192

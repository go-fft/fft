package fft

import "math"

// pow2StockhamMaxDefault routes every power of two to the Stockham engine on
// riscv64, factored with radix-4 passes and a lone radix-2 pass for an odd
// exponent (rule B, below). The passes run as Go code there, like the pow2
// kernel's. Measured with the scratch off dst's L1 sets (2026-10-04), at every
// 2^e from 256 to 2^20; time over the best of the pow2 kernel and three
// Stockham rules, geometric mean (worst):
//
//	                         pow2 kernel    Stockham, rule B
//	SiFive U74 (cfarm94)     1.156 (1.56)   1.127 (1.37)
//	SpacemiT X60 (cfarm95)   1.238 (1.53)   1.060 (1.24)
//
// On the U74 Stockham wins up to 32768 and the pow2 kernel above it (Stockham
// takes 1.03–1.27× its time from 65536 on); on the X60 Stockham with rule B
// wins at almost every size. Every Stockham policy tried scored 1.09–1.10 over
// both, against 1.196 for the pow2 kernel; rule B everywhere is the simplest
// of them. (The X60 host had four of its eight cores busy; the run was pinned
// to an idle one.) Until then riscv64 kept the pow2 kernel, measured ahead of
// Stockham (0.82–0.96×) on 2026-09-29, before the scratch fix.
func pow2StockhamMaxDefault() int { return math.MaxInt }

// r8MaxPow2Default and pow2OneRadix8Max select rule B for every power of two:
// radix 8 as far as it goes (rule A) scored 1.654 over the best on the X60,
// with a worst case of 2.56.
func r8MaxPow2Default() int { return 0 }

const pow2OneRadix8Max = 0

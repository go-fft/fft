package fft

// pow2StockhamMaxDefault routes powers of two up to 65536 to the Stockham
// engine on riscv64, factored with radix-4 passes and a lone radix-2 pass for
// an odd exponent (rule B, below), and larger ones to the pow2 kernel. The
// passes run as Go code there, like the pow2 kernel's. Measured with the
// scratch off dst's L1 sets (2026-10-04), at every 2^e from 256 to 2^20;
// pow2 kernel time ÷ Stockham rule B time:
//
//	n                        256   1024  4096  16384  65536  2^17  2^18  2^20
//	SiFive U74 (cfarm94)     1.34  1.06  1.18  1.14   0.97   0.89  0.79  0.82
//	SpacemiT X60 (cfarm95)   1.22  0.81  1.30  1.21   1.45   1.15  1.07  1.17
//
// Up to 65536 Stockham wins on both, but for the X60 at 1024. Above it the two
// disagree, so the pow2 kernel keeps them. End to end against main, the U74
// (idle) gained 1.04–1.34 on its real, inverse and 2-D rows up to 65536 and lost
// 0.80–0.92 at 2^20, which this limit avoids. (The X60 host had four of its
// eight cores busy; its runs were pinned to an idle one and are noisier.)
// Until then riscv64 kept the pow2 kernel throughout, measured ahead of
// Stockham (0.82–0.96×) on 2026-09-29, before the scratch fix.
func pow2StockhamMaxDefault() int { return 65536 }

// r8MaxPow2Default and pow2OneRadix8Max select rule B: radix 8 as far as it
// goes (rule A) took up to 2.56× the best rule's time on the X60.
func r8MaxPow2Default() int { return 0 }

const pow2OneRadix8Max = 0

// oddRadicesFirstDefault keeps the pocketfft order here (powers of two first,
// odd primes last): the odd-first order was measured on amd64 only (see
// route_amd64.go).
func oddRadicesFirstDefault() bool { return false }

// parMinChunkDefault is the 8192 elements chosen in 2026-09 (BENCHMARKS.md,
// Round 3), kept here: the change of Round 17 was measured on amd64 only.
func parMinChunkDefault() int { return 1 << 13 }

// parThresholdNDefault is parThreshold: the change of Round 21 was measured
// on arm64 only.
func parThresholdNDefault() int { return parThreshold }

// cascadeMinDefault is 0: the blocked schedule (cascade.go) was measured on
// amd64 only, so every length runs breadth first here.
func cascadeMinDefault() int { return 0 }

// znTwoPassMaxDefault and znThreePassMaxDefault are 0: kernels.ZnTwoPass and
// ZnThreePass are amd64's (Round 32).
func znTwoPassMaxDefault() int   { return 0 }
func znThreePassMaxDefault() int { return 0 }

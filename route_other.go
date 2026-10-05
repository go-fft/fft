//go:build !arm64 && !amd64 && !ppc64le && !riscv64

package fft

// pow2StockhamMaxDefault keeps powers of two on the iterative pow2 kernel on
// loong64 and s390x. On loong64 (3A5000) the two tied below 65536 and
// Stockham won only at 65536 (2026-09-29, before the scratch buffer was placed
// off dst's L1 sets, see offTheSets); its only host has been saturated since
// (load ~146 on 32 cores, 2026-10-04), so it was not re-measured. s390x was not
// reachable. ppc64le and riscv64 have their own routes.
func pow2StockhamMaxDefault() int { return 0 }

// r8MaxPow2Default: powers of two do not reach the Stockham engine here (they
// take the iterative pow2 kernel), so this only matters for a Plan built
// directly; it keeps the rule the engine was first calibrated with.
func r8MaxPow2Default() int { return 4096 }

// pow2OneRadix8Max: see r8MaxPow2Default.
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

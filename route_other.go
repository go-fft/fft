//go:build !arm64 && !amd64 && !ppc64le

package fft

// pow2StockhamMaxDefault keeps powers of two on the iterative pow2 kernel. On
// riscv64 (SpacemiT X60) that kernel was ahead of the Stockham engine
// (Stockham 0.82–0.96×); on loong64 (3A5000) the two tied below 65536 and
// Stockham won only at 65536. Measured on real hardware, 2026-09-29, before
// the scratch buffer was placed off dst's L1 sets (see offTheSets), which
// moved ppc64le to its own route; riscv64 and loong64 could not be re-measured
// (both hosts loaded, 2026-10-04), and s390x was not reachable.
func pow2StockhamMaxDefault() int { return 0 }

// r8MaxPow2Default: powers of two do not reach the Stockham engine here (they
// take the iterative pow2 kernel), so this only matters for a Plan built
// directly; it keeps the rule the engine was first calibrated with.
func r8MaxPow2Default() int { return 4096 }

// pow2OneRadix8Max: see r8MaxPow2Default.
const pow2OneRadix8Max = 0

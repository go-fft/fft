//go:build !arm64 && !amd64

package fft

// pow2StockhamMaxDefault keeps powers of two on the iterative pow2 kernel. On
// riscv64 (SpacemiT X60) that kernel is ahead of the Stockham engine (Stockham
// 0.82–0.96×); on ppc64le (POWER9) and loong64 (3A5000) the two tie below 65536
// and Stockham wins only at 65536, which does not justify a second route.
// Measured on real hardware, 2026-09-29; s390x was not reachable and keeps the
// pow2 kernel.
func pow2StockhamMaxDefault() int { return 0 }

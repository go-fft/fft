//go:build !arm64

package fft

// pow2StockhamDefault keeps powers of two on the iterative pow2 kernel. On
// amd64 that kernel's AVX2 butterflies run 2.2× faster than the Stockham
// engine (Xeon E5-2620 v3); on riscv64 (SpacemiT X60) the pow2 kernel is also
// ahead (Stockham 0.82–0.96×); on ppc64le (POWER9) and loong64 (3A5000) the two
// tie below 65536 and Stockham wins only at 65536, which does not justify a
// second route. Measured on real hardware, 2026-09-29; s390x was not reachable
// and keeps the pow2 kernel.
const pow2StockhamDefault = false

package kernels

// CMul computes the elementwise complex product a[i] *= b[i] over the first
// min(len(a), len(b)) elements, in place in a. It is the function the FFT core
// (Bluestein's pointwise spectral product) calls, and the stable seam behind
// which a SIMD kernel may be selected.
//
// Today CMul routes to the portable scalar oracle CMulScalar. Each supported
// architecture also ships a generated SIMD kernel (cmulSIMD, from
// cmul_<arch>.s) that the per-arch CI execution jobs assert is bit-identical to
// CMulScalar — but the kernel is NOT installed on the hot path, because the Go
// compiler's autovectorized scalar loop already matches or beats it at the
// widths Bluestein uses (measured: ~3x faster than the de-interleaving NEON
// kernel on arm64; see docs/plan-fft.md). Routing the hot path through a kernel
// we have measured to be slower would be a regression, so the scalar default
// stands until a kernel is measured to win. The SIMD work remains a validated,
// per-arch-tested artifact and the reference for future widening.
//
// Coverage policy. The portable scalar core (this file, cmul.go, kernels.go) is
// held to 100% statement coverage under the pure-Go gate. The generated
// assembly carries no Go statements; the per-arch cmulSIMD wrappers are reached
// only on their own architecture and are exercised by that architecture's
// native/qemu execution job (which runs the SIMD==scalar assertion), not by the
// single-run gate. Nothing is excluded from the gate that is not exercised by a
// per-arch job, and the gate is never lowered.
func CMul(a, b []complex128) {
	CMulScalar(a, b)
}

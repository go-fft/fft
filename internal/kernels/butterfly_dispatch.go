package kernels

// Radix2Stage and Radix4Stage are the functions the iterative FFT core calls for
// one decimation-in-time pass; they are the stable seam behind which a per-arch
// SIMD kernel is selected.
//
// Unlike CMul (whose scalar default was measured to beat its SIMD kernel at the
// short Bluestein widths), these stage kernels are routed onto the SIMD path on
// the arch that was measured to win (amd64, where SSE2 has real packed ADDPD/
// SUBPD and the GOAMD64=v1 autovectorizer is weakest — see UseSIMDButterfly).
// The bit-identity contract is the same as cmul's — each arch's kernel is
// asserted bit-for-bit identical to the scalar oracle by that arch's CI
// execution job — so routing here cannot change a transform's result.
//
// radix2StageSIMD / radix4StageSIMD are provided per arch (butterfly_<arch>.go)
// and alias the scalar oracle on arches without a kernel (butterfly_generic.go),
// so these dispatchers are always safe to call.

// Radix2Stage runs one radix-2 DIT pass: see Radix2StageScalar.
func Radix2Stage(a []complex128, n, span int, tw []complex128) {
	if span == 0 || n == 0 {
		return
	}
	radix2StageSIMD(a, n, span, tw)
}

// Radix4Stage runs one radix-4 DIT pass: see Radix4StageScalar.
func Radix4Stage(a []complex128, n, span int, w1, w2, w3 []complex128, inverse bool) {
	if span == 0 || n == 0 {
		return
	}
	radix4StageSIMD(a, n, span, w1, w2, w3, inverse)
}

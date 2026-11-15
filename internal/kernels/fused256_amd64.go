package kernels

// Complex128 256-point FFT as two fused 16x16 AVX-512 passes (go-fft/fft#124).
// fused256_amd64.s and fused256_tw_amd64.go (the twiddle table, correctly
// rounded) are written by internal/genfused256.

// fused256 transforms 256 points (both directions) and reports whether it did.
// Its entry is a frame-free NOSPLIT function whose first instructions read
// Fused256On and UseStockhamAVX512, at every call, and return false before any
// AVX-512 instruction when either is off; otherwise it jumps to fused256body,
// which sets up the 4160-byte frame. The caller then runs its own passes
// (ZnPlan.Run's pattern). tw is the forward table; the inverse one follows it
// in memory (fused256Tw is one array).
//
//go:noescape
func fused256(dst, src *[256]complex128, tw *float64, inv bool) (ok bool)

// fused256body is fused256's body, reached only by the jump (vet wants it
// declared).
//
//go:noescape
func fused256body(dst, src *[256]complex128, tw *float64, inv bool) (ok bool)

// fused256ref is the same transform without the add-to-FMA rewrite (and without
// the gates): the schedule of v0.34.0, kept for the tests that assert the two
// give the same bits. Nothing else calls it, so the linker drops it.
//
//go:noescape
func fused256ref(dst, src *complex128, tw *float64, inv bool) (ok bool)

// Fused256Ref runs the reference schedule (fused256ref). For tests; it needs
// AVX-512 and does not check for it.
func Fused256Ref(dst, src *[256]complex128, inverse bool) bool {
	return fused256ref(&dst[0], &src[0], fused256TwP, inverse)
}

// Fused256Table is the twiddle table of one direction, for the tests that
// perturb it to show what the accuracy bound can see.
func Fused256Table(inverse bool) []float64 {
	Fused256Prepare()
	if inverse {
		return fused256TwA[len(fused256TwA)/2:]
	}
	return fused256TwA[:len(fused256TwA)/2]
}

// Fused256On is whether the fused kernel runs here: AVX-512 (F and DQ, the
// probe of UseStockhamAVX512) on an AMD Zen 4 (go-fft/fft#124) or an Intel
// Cascade Lake (IntelCascadeLake, 0.82 of the plan's time there, go-fft/fft#162);
// no other CPU was measured. The kernel reads it, and UseStockhamAVX512, at
// every call.
var Fused256On = UseStockhamAVX512 && (AMDZen4 || IntelCascadeLake)

// Fused256 transforms the 256 points of src into dst (which may alias
// src; inverse is unnormalized) and reports whether the kernel ran: false, with
// dst untouched, when Fused256On or UseStockhamAVX512 is off. The arguments are
// array pointers (a caller converts its slices, which panics on a short one), so
// that it is small enough to inline.
func Fused256(dst, src *[256]complex128, inverse bool) bool {
	return fused256(dst, src, fused256TwP, inverse)
}

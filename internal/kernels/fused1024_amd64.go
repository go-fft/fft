package kernels

// Complex128 1024-point FFT as two fused 32x32 AVX-512 passes (go-fft/fft#162).
// fused1024_amd64.s and fused1024_tw_amd64.go (the twiddle table, correctly
// rounded) are written by internal/genfused256.

// fused1024 transforms 1024 points (both directions) and reports whether it did.
// Its entry is a frame-free NOSPLIT function whose first instructions read
// Fused1024On and UseStockhamAVX512, at every call, and return false before any
// AVX-512 instruction when either is off; otherwise it jumps to fused1024body,
// which sets up the frame. The caller then runs its own passes. tw is the
// forward table; the inverse one follows it in memory (fused1024Tw is one array).
//
//go:noescape
func fused1024(dst, src *[1024]complex128, tw *float64, inv bool) (ok bool)

// fused1024body is fused1024's body, reached only by the jump (vet wants it
// declared).
//
//go:noescape
func fused1024body(dst, src *[1024]complex128, tw *float64, inv bool) (ok bool)

// Fused1024Table is the twiddle table of one direction, for the tests that
// perturb it to show what the accuracy bound can see.
func Fused1024Table(inverse bool) []float64 {
	if inverse {
		return fused1024TwA[len(fused1024TwA)/2:]
	}
	return fused1024TwA[:len(fused1024TwA)/2]
}

// Fused1024On is whether the fused 1024-point kernel runs here: AVX-512 (F and
// DQ, the probe of UseStockhamAVX512) on an AMD Zen 4, the only CPU it was
// measured on (go-fft/fft#162). The kernel reads it, and UseStockhamAVX512, at
// every call.
var Fused1024On = UseStockhamAVX512 && AMDZen4

// Fused1024 transforms the 1024 points of src into dst (which may alias src;
// inverse is unnormalized) and reports whether the kernel ran: false, with dst
// untouched, when Fused1024On or UseStockhamAVX512 is off.
func Fused1024(dst, src *[1024]complex128, inverse bool) bool {
	return fused1024(dst, src, fused1024TwP, inverse)
}

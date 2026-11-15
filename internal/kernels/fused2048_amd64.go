package kernels

// Complex128 2048-point FFT as two fused AVX-512 passes (go-fft/fft#162).
// fused2048_amd64.s and fused2048_tw_amd64.go (the twiddle table, correctly
// rounded) are written by internal/genfused256.

// fused2048 transforms 2048 points (both directions) and reports whether it did.
// Its entry is a frame-free NOSPLIT function whose first instructions read
// Fused2048On and UseStockhamAVX512, at every call, and return false before any
// AVX-512 instruction when either is off; otherwise it jumps to fused2048body,
// which sets up the frame. The caller then runs its own passes. tw is the
// forward table; the inverse one follows it (fused2048TwA is one block).
//
//go:noescape
func fused2048(dst, src *[2048]complex128, tw *float64, inv bool) (ok bool)

// fused2048body is fused2048's body, reached only by the jump (vet wants it
// declared).
//
//go:noescape
func fused2048body(dst, src *[2048]complex128, tw *float64, inv bool) (ok bool)

// Fused2048Table is the twiddle table of one direction (the 64-byte aligned copy
// the kernel reads), for the tests that perturb it to show what the accuracy
// bound can see.
func Fused2048Table(inverse bool) []float64 {
	Fused2048Prepare()
	if inverse {
		return fused2048TwA[len(fused2048TwA)/2:]
	}
	return fused2048TwA[:len(fused2048TwA)/2]
}

// Fused2048On is whether the fused 2048-point kernel runs here: AVX-512 (F and
// DQ, the probe of UseStockhamAVX512) on an AMD Zen 4 or an Intel Cascade Lake
// (IntelCascadeLake, 0.79 of the plan's time there), the CPUs it was measured on
// (go-fft/fft#162). The kernel reads it, and UseStockhamAVX512, at every call.
var Fused2048On = UseStockhamAVX512 && (AMDZen4 || IntelCascadeLake)

// Fused2048 transforms the 2048 points of src into dst (which may alias src;
// inverse is unnormalized) and reports whether the kernel ran: false, with dst
// untouched, when Fused2048On or UseStockhamAVX512 is off.
func Fused2048(dst, src *[2048]complex128, inverse bool) bool {
	return fused2048(dst, src, fused2048TwP, inverse)
}

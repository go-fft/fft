package fft

import "github.com/go-fft/fft/internal/kernels"

// fusedSRun transforms the n points with the fused complex64 kernel of that
// length and reports whether it did (Plan32.execute falls back to its passes on
// false: the kernel refuses on a CPU, or while a test has the AVX-512 flag off).
// The slices convert to array pointers, which panics on a short one.
func fusedSRun(n int, dst, src []complex64, inverse bool) bool {
	switch n {
	case 64:
		return kernels.FusedS64((*[64]complex64)(dst), (*[64]complex64)(src), inverse)
	case 128:
		return kernels.FusedS128((*[128]complex64)(dst), (*[128]complex64)(src), inverse)
	case 256:
		return kernels.FusedS256((*[256]complex64)(dst), (*[256]complex64)(src), inverse)
	case 512:
		return kernels.FusedS512((*[512]complex64)(dst), (*[512]complex64)(src), inverse)
	case 1024:
		return kernels.FusedS1024((*[1024]complex64)(dst), (*[1024]complex64)(src), inverse)
	case 2048:
		return kernels.FusedS2048((*[2048]complex64)(dst), (*[2048]complex64)(src), inverse)
	case 4096:
		return kernels.FusedS4096((*[4096]complex64)(dst), (*[4096]complex64)(src), inverse)
	case 8192:
		return kernels.FusedS8192((*[8192]complex64)(dst), (*[8192]complex64)(src), inverse)
	}
	return false
}

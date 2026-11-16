package fft

import "github.com/go-fft/fft/internal/kernels"

// fusedS<n>Run transforms n points with the fused complex64 kernel of that length
// and reports whether it did (Plan32.execute falls back to its passes on false: the
// kernel refuses on a CPU, or while a test has the AVX-512 flag off). Each is a
// call the compiler inlines into execute (a single function with a switch over the
// lengths cost 596 and was a call of its own, go-fft/fft#165 for complex128). The
// slices convert to array pointers, which panics on a short one. The other
// platforms' twins report false.

func fusedS64Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS64((*[64]complex64)(dst), (*[64]complex64)(src), inverse)
}

func fusedS128Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS128((*[128]complex64)(dst), (*[128]complex64)(src), inverse)
}

func fusedS256Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS256((*[256]complex64)(dst), (*[256]complex64)(src), inverse)
}

func fusedS512Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS512((*[512]complex64)(dst), (*[512]complex64)(src), inverse)
}

func fusedS1024Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS1024((*[1024]complex64)(dst), (*[1024]complex64)(src), inverse)
}

func fusedS2048Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS2048((*[2048]complex64)(dst), (*[2048]complex64)(src), inverse)
}

func fusedS4096Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS4096((*[4096]complex64)(dst), (*[4096]complex64)(src), inverse)
}

func fusedS8192Run(dst, src []complex64, inverse bool) bool {
	return kernels.FusedS8192((*[8192]complex64)(dst), (*[8192]complex64)(src), inverse)
}

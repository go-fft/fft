//go:build !amd64

package fft

// The fused complex64 kernels are amd64's; here they never run.

func fusedS64Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS128Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS256Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS512Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS1024Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS2048Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS4096Run(dst, src []complex64, inverse bool) bool { return false }

func fusedS8192Run(dst, src []complex64, inverse bool) bool { return false }

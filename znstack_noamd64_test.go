//go:build !amd64

package fft

// znAVX2Route is a no-op where there are no AVX2 frame kernels: the plans
// keep this architecture's route (see the amd64 version).
func znAVX2Route() func() { return func() {} }

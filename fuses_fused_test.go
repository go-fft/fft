//go:build !amd64 || amd64.v3

package fft

// scalarFuses reports whether gc may fuse the scalar passes' multiply-adds.
// Here it does (FMA is baseline on arm64, ppc64le, s390x, riscv64 and
// loong64, and in GOAMD64=v3), and which product of a complex multiply it
// fuses depends on how the expression reached the compiler, so the Go
// batched pass and the scalar passes may differ in the last bit. The batched
// path only runs in production with the AVX2 kernels, which do not fuse, and
// the NEON ones, which fuse exactly what the 1-D passes fuse (stripsExact).
const scalarFuses = true

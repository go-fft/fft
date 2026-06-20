//go:build !riscv64

package kernels

import "testing"

// skipIfNoSIMD is a no-op on every architecture whose SIMD kernel is part of the
// baseline (amd64/arm64/s390x) or whose cmulSIMD aliases the scalar path
// (loong64/ppc64le): in all these cases cmulSIMD is always safe to call and the
// SIMD-vs-scalar comparison always runs. riscv64 overrides this (see
// cmul_skip_riscv64_test.go) because its V extension is optional and detected at
// run time.
func skipIfNoSIMD(*testing.T) {}

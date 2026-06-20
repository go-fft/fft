//go:build ignore

// Command gen produces cmul_arm64.s, the NEON pointwise complex multiply
// kernel, via go-asmgen. Run with: go run gen.go (or `go generate` from the
// kernels package).
//
// cmulNEON(a, b *complex128, n int) computes a[i] = a[i] * b[i] for i in [0,n).
// A complex128 is two contiguous float64 {re, im} (16 bytes). The kernel
// processes TWO complex128 per iteration with NEON: VLD2 deinterleaves a pair
// into a vector of reals and a vector of imaginaries, the arithmetic runs
// two-lane, and VST2 re-interleaves on store. A scalar tail handles an odd
// final element.
//
// Per element, with a = [ar, ai] and b = [br, bi], the product is
//
//	re = ar*br - ai*bi
//	im = ar*bi + ai*br
//
// Bit-for-bit identity with the scalar oracle CMulScalar is the contract, and
// it constrains the FUSION form. Unlike amd64 (whose SSE2 MULPD/ADDPD are
// separately rounded and whose GOAMD64=v1 oracle does NOT fuse), the gc
// compiler emits a FUSED form for the oracle on arm64, because FMA is baseline:
//
//	re = (ar*br) then FMSUBD  -> a single fused multiply-subtract for ai*bi
//	im = (ar*bi) then FMADDD  -> a single fused multiply-add for ai*br
//
// (verified by disassembling CMulScalar with -gcflags=-S). A NEON kernel is
// therefore bit-identical ONLY if it reproduces that same fusion:
//
//	re acc = 0; VFMLA br,ar (acc += ar*br); VFMLS bi,ai (acc -= ai*bi)
//	im acc = 0; VFMLA bi,ar (acc += ar*bi); VFMLA br,ai (acc += ai*br)
//
// VFMLA into a zeroed accumulator yields the exactly-rounded product (adding
// +0.0 does not change rounding), matching the oracle's leading FMULD; the
// trailing VFMLS / VFMLA are the fused subtract / add the oracle performs. A
// naive non-fused NEON (separate multiply then add) — or, equally, a fused
// kernel whose fusion form differs from the oracle's — diverges by up to 1 ULP;
// the random SIMD-vs-scalar test catches it, so this is validated, not assumed.
//
// Go's arm64 assembler exposes no non-fused vector float multiply (there is no
// VFMUL), so matching the oracle's fused form is the only way to vectorize this
// soundly. We do, and the per-arch arm64 CI job asserts bit identity.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")

	sig := arm64.Layout(
		[]string{"a", "b", "n"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
	b := arm64.NewFunc("cmulNEON", sig, 0)
	b.LoadArg("a", "R0").
		LoadArg("b", "R1").
		LoadArg("n", "R2").
		Raw("loop2:").
		Raw("CMP $2, R2").
		Raw("BLT tail").
		Raw("VLD2 (R0), [V0.D2, V1.D2]"). // V0=[ar0,ar1] V1=[ai0,ai1]
		Raw("VLD2 (R1), [V2.D2, V3.D2]"). // V2=[br0,br1] V3=[bi0,bi1]
		Raw("VEOR V4.B16, V4.B16, V4.B16"). // re acc = 0
		Raw("VEOR V5.B16, V5.B16, V5.B16"). // im acc = 0
		Raw("VFMLA V2.D2, V0.D2, V4.D2"). // re += ar*br
		Raw("VFMLS V3.D2, V1.D2, V4.D2"). // re -= ai*bi  (fused, matches oracle FMSUBD)
		Raw("VFMLA V3.D2, V0.D2, V5.D2"). // im += ar*bi
		Raw("VFMLA V2.D2, V1.D2, V5.D2"). // im += ai*br  (fused, matches oracle FMADDD)
		Raw("VST2 [V4.D2, V5.D2], (R0)").
		Raw("ADD $32, R0").
		Raw("ADD $32, R1").
		Raw("SUB $2, R2").
		Raw("B loop2").
		Raw("tail:").
		Raw("CBZ R2, done").
		Raw("FMOVD (R0), F0"). // ar
		Raw("FMOVD 8(R0), F1"). // ai
		Raw("FMOVD (R1), F2"). // br
		Raw("FMOVD 8(R1), F3"). // bi
		Raw("FMULD F2, F0, F4"). // ar*br
		Raw("FMSUBD F1, F4, F3, F4"). // F4 - ai*bi  (fused, matches oracle)
		Raw("FMULD F3, F0, F5"). // ar*bi
		Raw("FMADDD F2, F5, F1, F5"). // F5 + ai*br  (fused, matches oracle)
		Raw("FMOVD F4, (R0)").
		Raw("FMOVD F5, 8(R0)").
		Raw("done:").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("cmul_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmul_arm64.s")
}

//go:build ignore

// Command gen produces cmul_amd64.s, the SSE2 pointwise complex multiply
// kernel, via go-asmgen. Run with: go run gen.go (or `go generate` from the
// kernels package).
//
// cmulSSE2(a, b *complex128, n int) computes a[i] = a[i] * b[i] for i in [0,n).
// A complex128 is two contiguous float64 {re, im} (16 bytes). SSE2 packed
// double lets us process one complex128 per loop iteration with no horizontal
// (SSE3) instruction, so the kernel runs on the entire amd64 baseline without
// any CPU-feature detection.
//
// Per element, with a = [ar, ai] and b = [br, bi], the product is
//
//	re = ar*br - ai*bi
//	im = ar*bi + ai*br
//
// computed as (broadcast + swap + sign-flip, all SSE2):
//
//	are = [ar, ar]          (SHUFPD $0  of a)
//	aim = [ai, ai]          (SHUFPD $3  of a)
//	t0  = are * b   = [ar*br, ar*bi]
//	bsw = [bi, br]          (SHUFPD $1  of b)
//	t1  = aim * bsw = [ai*bi, ai*br]
//	t1 ^= [sign, 0] = [-ai*bi, ai*br]   (XORPD with low-lane sign mask)
//	res = t0 + t1   = [ar*br - ai*bi, ar*bi + ai*br]
//
// The low-lane sign mask [sign, 0] is built once in a register with pure SSE2
// (PCMPEQL all-ones -> PSLLQ $63 gives [sign, sign]; MOVSD into a zeroed
// register keeps only the low lane), so the kernel needs no read-only data
// table.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")

	sig := amd64.Layout(
		[]string{"a", "b", "n"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc("cmulSSE2", sig, 0)
	b.LoadArg("a", "AX").
		LoadArg("b", "BX").
		LoadArg("n", "CX").
		// Build the low-lane sign mask X7 = [0x8000000000000000, 0].
		Raw("XORPS X6, X6").       // X6 = [0, 0]
		Raw("PCMPEQL X7, X7").     // X7 = all ones
		Raw("PSLLQ $63, X7").      // X7 = [sign, sign]
		Raw("MOVSD X7, X6").       // X6 = [sign, 0]  (low lane only)
		Raw("MOVAPS X6, X7").      // X7 = [sign, 0]
		Raw("loop:").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Raw("MOVUPD (AX), X0").    // X0 = a = [ar, ai]
		Raw("MOVUPD (BX), X1").    // X1 = b = [br, bi]
		Raw("MOVAPS X0, X2").
		Raw("SHUFPD $0, X2, X2").  // X2 = [ar, ar]
		Raw("MOVAPS X0, X3").
		Raw("SHUFPD $3, X3, X3").  // X3 = [ai, ai]
		Raw("MOVAPS X1, X4").
		Raw("SHUFPD $1, X4, X4").  // X4 = [bi, br]
		Raw("MULPD X1, X2").       // X2 = [ar*br, ar*bi]
		Raw("MULPD X4, X3").       // X3 = [ai*bi, ai*br]
		Raw("XORPD X7, X3").       // X3 = [-ai*bi, ai*br]
		Raw("ADDPD X3, X2").       // X2 = [ar*br-ai*bi, ar*bi+ai*br]
		Raw("MOVUPD X2, (AX)").    // a[i] = product
		Raw("ADDQ $16, AX").
		Raw("ADDQ $16, BX").
		Raw("DECQ CX").
		Raw("JMP loop").
		Raw("done:").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("cmul_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmul_amd64.s")
}

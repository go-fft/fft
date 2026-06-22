//go:build ignore

// Command gen produces, via go-asmgen:
//   - cmul_amd64.s      : SSE2 pointwise complex multiply (a[i] *= b[i]).
//   - butterfly_amd64.s : SSE2 radix-2 and radix-4 decimation-in-time butterfly
//     STAGE kernels — the FFT hot loop (the whole pass, both the loop over groups
//     and the loop over butterfly positions, runs inside one call).
//
// Run with: go run gen.go (or `go generate` from the kernels package).
//
// A complex128 is two contiguous float64 {re, im} (16 bytes). SSE2 packed double
// processes one complex128 per register with no horizontal (SSE3) instruction,
// so these kernels run on the entire amd64 baseline with no CPU-feature branch.
// The packed ADDPD/SUBPD do re and im in one instruction (2× the scalar amd64
// throughput, since GOAMD64=v1 Go does not autovectorize the scalar loop), and
// MULPD/ADDPD are SEPARATELY rounded — matching the non-fused GOAMD64=v1 scalar
// oracle bit-for-bit (no FMA, so no 1-ULP divergence). The complex product uses
// the same broadcast+swap+sign-mask sequence as cmul_amd64.s.
//
// Per element a=[ar,ai], b=[br,bi]: re=ar·br−ai·bi, im=ar·bi+ai·br, computed as
//
//	are = [ar,ar] (SHUFPD $0); aim = [ai,ai] (SHUFPD $3)
//	bsw = [bi,br] (SHUFPD $1)
//	t0  = are*b = [ar·br, ar·bi]; t1 = aim*bsw = [ai·bi, ai·br]
//	t1 ^= [sign,0] -> [−ai·bi, ai·br]; res = t0+t1 = [re, im]
//
// The low-lane sign mask [sign,0] is built once with pure SSE2 (PCMPEQL all-ones
// -> PSLLQ $63 -> [sign,sign]; MOVSD into a zeroed reg keeps the low lane), so
// the kernels need no read-only data table.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

// signMask emits the SSE2 sequence building X7 = [0x8000000000000000, 0], the
// low-lane sign flip the complex product uses.
func signMask(b *amd64.Builder) {
	b.Raw("XORPS X6, X6"). // X6 = [0,0]
				Raw("PCMPEQL X7, X7"). // X7 = all ones
				Raw("PSLLQ $63, X7").  // X7 = [sign,sign]
				Raw("MOVSD X7, X6").   // X6 = [sign,0]
				Raw("MOVAPS X6, X7")   // X7 = [sign,0]
}

// scmul emits dst = (xPtr)·(yPtr) into register dst, the SSE2 complex product.
// Uses X2,X3,X4 as scratch; X7 must hold the sign mask. dst is one of X0,X1,X5.
func scmul(b *amd64.Builder, xPtr, yPtr, dst string) {
	b.Raw("MOVUPD (" + xPtr + "), X0"). // a = [ar,ai]
						Raw("MOVUPD (" + yPtr + "), X1"). // b = [br,bi]
						Raw("MOVAPS X0, X2").
						Raw("SHUFPD $0, X2, X2"). // [ar,ar]
						Raw("MOVAPS X0, X3").
						Raw("SHUFPD $3, X3, X3"). // [ai,ai]
						Raw("MOVAPS X1, X4").
						Raw("SHUFPD $1, X4, X4"). // [bi,br]
						Raw("MULPD X1, X2").      // [ar*br, ar*bi]
						Raw("MULPD X4, X3").      // [ai*bi, ai*br]
						Raw("XORPD X7, X3").      // [-ai*bi, ai*br]
						Raw("ADDPD X3, X2")       // [re, im]
	if dst != "X2" {
		b.Raw("MOVAPS X2, " + dst)
	}
}

func genCmul(f *emit.File) {
	sig := amd64.Layout(
		[]string{"a", "b", "n"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc("cmulSSE2", sig, 0)
	b.LoadArg("a", "AX").
		LoadArg("b", "BX").
		LoadArg("n", "CX")
	signMask(b)
	b.Raw("loop:").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Raw("MOVUPD (AX), X0").
		Raw("MOVUPD (BX), X1").
		Raw("MOVAPS X0, X2").
		Raw("SHUFPD $0, X2, X2").
		Raw("MOVAPS X0, X3").
		Raw("SHUFPD $3, X3, X3").
		Raw("MOVAPS X1, X4").
		Raw("SHUFPD $1, X4, X4").
		Raw("MULPD X1, X2").
		Raw("MULPD X4, X3").
		Raw("XORPD X7, X3").
		Raw("ADDPD X3, X2").
		Raw("MOVUPD X2, (AX)").
		Raw("ADDQ $16, AX").
		Raw("ADDQ $16, BX").
		Raw("DECQ CX").
		Raw("JMP loop").
		Raw("done:").
		Ret()
	f.Add(b.Func())
}

// genRadix2 emits radix2StageSSE2(a *complex128, n, span int, tw *complex128):
//
//	for base := 0; base < n; base += 2*span {
//	  for k := 0; k < span; k++ {
//	    t = a[base+span+k] * tw[k]
//	    a[base+k], a[base+span+k] = a[base+k]+t, a[base+k]-t
//	  }
//	}
//
// Registers: SI=&a[base] (lo ptr), DI=&a[base+span] (hi ptr), R8=&tw[0],
// R9=tw cursor, R10=k counter, R11=span, R12=base, AX=&a[0], CX=n,
// DX=span*16 (span bytes), R13=base step (2*span*16). X7=sign mask.
func genRadix2(f *emit.File) {
	sig := amd64.Layout(
		[]string{"a", "n", "span", "tw"},
		[]amd64.Type{amd64.Ptr, amd64.Int64, amd64.Int64, amd64.Ptr},
		nil, nil,
	)
	b := amd64.NewFunc("radix2StageSSE2", sig, 0)
	b.LoadArg("a", "AX").
		LoadArg("n", "CX").
		LoadArg("span", "R11").
		LoadArg("tw", "R8")
	signMask(b)
	b.Raw("MOVQ R11, DX").
		Raw("SHLQ $4, DX"). // DX = span*16 (byte offset lo->hi, and tw stride)
		Raw("MOVQ DX, R13").
		Raw("SHLQ $1, R13"). // R13 = 2*span*16 (group byte stride)
		Raw("MOVQ CX, R14").
		Raw("SHLQ $4, R14").  // R14 = n*16 (end byte offset)
		Raw("XORQ R12, R12"). // R12 = base byte offset = 0
		Raw("baseloop:").
		Raw("CMPQ R12, R14").
		Raw("JGE done").
		Raw("LEAQ (AX)(R12*1), SI"). // SI = &a[base]
		Raw("LEAQ (SI)(DX*1), DI").  // DI = &a[base+span]
		Raw("MOVQ R8, R9").          // R9 = &tw[0]
		Raw("XORQ R10, R10").        // k = 0
		Raw("kloop:").
		Raw("CMPQ R10, R11").
		Raw("JGE basenext")
	scmul(b, "DI", "R9", "X5") // X5 = t = hi*tw
	b.Raw("MOVUPD (SI), X0").  // X0 = lo
					Raw("MOVAPS X0, X1").
					Raw("ADDPD X5, X0"). // lo+t
					Raw("SUBPD X5, X1"). // lo-t
					Raw("MOVUPD X0, (SI)").
					Raw("MOVUPD X1, (DI)").
					Raw("ADDQ $16, SI").
					Raw("ADDQ $16, DI").
					Raw("ADDQ $16, R9").
					Raw("INCQ R10").
					Raw("JMP kloop").
					Raw("basenext:").
					Raw("ADDQ R13, R12").
					Raw("JMP baseloop").
					Raw("done:").
					Ret()
	f.Add(b.Func())
}

// genRadix4 emits radix4StageSSE2{Fwd,Inv}(a *complex128, n, span int,
// w1,w2,w3 *complex128). The ∓i rotation is a SHUFPD lane swap plus an XORPD
// sign flip of one lane (exact). Two variants avoid an inner branch on inverse.
//
// Register map (group loop):
//
//	AX = base byte offset (advances by 4*span*16 per group)
//	CX = n*16 (end, constant)         R15 = span*16 (block stride, constant)
//	R11 = 4*span*16 (group stride)     R14 = k counter
//	SI,DI,R12,R13 = b0,b1,b2,b3 cursors
//	R8,R9,R10 = w1,w2,w3 cursors (reloaded from args each group)
//	X7 = sign mask
//
// The a base and the three twiddle bases are reloaded from the FP frame each
// group (LoadArg), which restarts the twiddle cursors and needs no extra GP reg.
func genRadix4(f *emit.File, name string, inverse bool) {
	sig := amd64.Layout(
		[]string{"a", "n", "span", "w1", "w2", "w3"},
		[]amd64.Type{amd64.Ptr, amd64.Int64, amd64.Int64, amd64.Ptr, amd64.Ptr, amd64.Ptr},
		nil, nil,
	)
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("n", "CX").
		LoadArg("span", "R15")
	signMask(b)
	b.Raw("MOVQ R15, R11").
		Raw("SHLQ $6, R11"). // R11 = 4*span*16 (group stride)
		Raw("SHLQ $4, CX").  // CX = n*16 (end)
		Raw("SHLQ $4, R15"). // R15 = span*16 (block stride)
		Raw("XORQ AX, AX").  // base byte offset = 0
		Raw("baseloop:").
		Raw("CMPQ AX, CX").
		Raw("JGE done")
	// Operand cursors from a base + base offset.
	b.LoadArg("a", "SI").
		Raw("ADDQ AX, SI").          // SI = &a[base]
		Raw("LEAQ (SI)(R15*1), DI"). // b1
		Raw("LEAQ (DI)(R15*1), R12").
		Raw("LEAQ (R12)(R15*1), R13")
	// Twiddle cursors restart at the plane base each group.
	b.LoadArg("w1", "R8").
		LoadArg("w2", "R9").
		LoadArg("w3", "R10").
		Raw("XORQ R14, R14"). // k = 0
		Raw("kloop:").
		Raw("CMPQ R14, R15").
		Raw("JGE basenext") // compare k*16 against span*16? we step k by 16 bytes below
	// m1 = b1*w1 -> X1 ; m2 = b2*w2 -> X5 ; m3 = b3*w3 -> need a 3rd. We compute in
	// order, holding intermediates on the stack-free X regs X0,X1,X5 and recomputing
	// combinations. SSE has X0..X15; use high regs to avoid scmul's X0..X4 scratch.
	scmul(b, "DI", "R8", "X8")    // m1
	scmul(b, "R12", "R9", "X9")   // m2
	scmul(b, "R13", "R10", "X10") // m3
	b.Raw("MOVUPD (SI), X11").    // a0 = b0[k]
					Raw("MOVAPS X11, X12").
					Raw("ADDPD X9, X11"). // t0 = a0 + m2
					Raw("SUBPD X9, X12"). // t1 = a0 - m2
					Raw("MOVAPS X8, X13").
					Raw("MOVAPS X8, X14").
					Raw("ADDPD X10, X13"). // t2 = m1 + m3
					Raw("SUBPD X10, X14")  // d  = m1 - m3
	// t3 = rot(d). d = [dre, dim]. forward t3 = [dim, -dre]; inverse t3 = [-dim, dre].
	b.Raw("MOVAPS X14, X15").
		Raw("SHUFPD $1, X15, X15") // X15 = [dim, dre]
	if inverse {
		// t3 = [-dim, dre]: flip low lane sign.
		b.Raw("XORPD X7, X15") // [-dim, dre]
	} else {
		// t3 = [dim, -dre]: flip high lane sign. Build a high-lane mask in X6? X6 is
		// [0,0] from signMask; rebuild [0,sign] via shuffling X7=[sign,0].
		b.Raw("MOVAPS X7, X6").
			Raw("SHUFPD $1, X6, X6"). // X6 = [0, sign]
			Raw("XORPD X6, X15")      // [dim, -dre]
	}
	// b0 = t0+t2 ; b2 = t0-t2 ; b1 = t1+t3 ; b3 = t1-t3.
	b.Raw("MOVAPS X11, X0").
		Raw("ADDPD X13, X0").
		Raw("MOVUPD X0, (SI)"). // b0
		Raw("MOVAPS X11, X0").
		Raw("SUBPD X13, X0").
		Raw("MOVUPD X0, (R12)"). // b2
		Raw("MOVAPS X12, X0").
		Raw("ADDPD X15, X0").
		Raw("MOVUPD X0, (DI)"). // b1
		Raw("MOVAPS X12, X0").
		Raw("SUBPD X15, X0").
		Raw("MOVUPD X0, (R13)"). // b3
		Raw("ADDQ $16, SI").
		Raw("ADDQ $16, DI").
		Raw("ADDQ $16, R12").
		Raw("ADDQ $16, R13").
		Raw("ADDQ $16, R8").
		Raw("ADDQ $16, R9").
		Raw("ADDQ $16, R10").
		Raw("ADDQ $16, R14").
		Raw("JMP kloop").
		Raw("basenext:").
		Raw("ADDQ R11, AX").
		Raw("JMP baseloop").
		Raw("done:").
		Ret()
	f.Add(b.Func())
}

func main() {
	fc := emit.NewFile("amd64")
	genCmul(fc)
	writeFile("cmul_amd64.s", fc.String())

	fb := emit.NewFile("amd64")
	genRadix2(fb)
	genRadix4(fb, "radix4StageSSE2Fwd", false)
	genRadix4(fb, "radix4StageSSE2Inv", true)
	writeFile("butterfly_amd64.s", fb.String())
}

func writeFile(name, content string) {
	if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

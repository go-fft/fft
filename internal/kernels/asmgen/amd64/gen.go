//go:build ignore

// Command gen produces, via go-asmgen:
//   - cmul_amd64.s      : SSE2 pointwise complex multiply (a[i] *= b[i]).
//   - butterfly_amd64.s : SSE2/AVX2 radix-2 and radix-4 decimation-in-time butterfly
//     STAGE kernels — the FFT hot loop (the whole pass, both the loop over groups
//     and the loop over butterfly positions, runs inside one call).
//
// Run with: go run gen.go (or `go generate` from the kernels package).
//
// A complex128 is two contiguous float64 {re, im} (16 bytes). SSE2 packed double
// processes one complex128 per register with no horizontal (SSE3) instruction,
// so these kernels run on the entire amd64 baseline with no CPU-feature branch.
// AVX2 variants below process two independent butterflies in each YMM register;
// the Go wrapper checks CPU/OS support and shape before selecting them.
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
	genRadix4AVX2(fb, "radix4StageAVX2Fwd", false)
	genRadix4AVX2(fb, "radix4StageAVX2Inv", true)
	genRadix2LeafAVX2(fb)
	for _, r := range []int{2, 3, 4, 5, 8} {
		genStockhamAVX2(fb, r)
	}
	for _, r := range []int{2, 3, 4, 5, 8} {
		genStockhamLastAVX2(fb, r)
	}
	writeFile("butterfly_amd64.s", fb.String())

	// The gate in front of those AVX2 kernels. It used to be 24 hand-written
	// lines in cpu_amd64.s; go-asmgen v0.9.0 emits the same sequence, and the
	// OS half -- OSXSAVE, then XGETBV's XMM and YMM bits in XCR0, before the
	// feature bit -- is the part a hand-rolled probe drops. A CPU can report
	// AVX2 on a kernel that does not save YMM state, and then the upper lanes
	// go at the next context switch.
	fp := emit.NewFile("amd64")
	fp.Add(amd64.FeatureProbe("supportsAVX2", amd64.AVX2))
	writeFile("cpu_amd64.s", fp.String())
}

// genRadix2LeafAVX2 pairs two span-one groups. Keep the twiddle multiply even
// when it is unity: eliding it would change signed-zero/non-finite behavior.
func genRadix2LeafAVX2(f *emit.File) {
	sig := amd64.Layout([]string{"a", "n", "tw"},
		[]amd64.Type{amd64.Ptr, amd64.Int64, amd64.Ptr}, nil, nil)
	b := amd64.NewFunc("radix2LeafAVX2", sig, 0)
	b.LoadArg("a", "AX").LoadArg("n", "CX").LoadArg("tw", "DX")
	b.Raw("MOVQ $0x8000000000000000, BX").
		Raw("VMOVQ BX, X7").Raw("VINSERTF128 $1, X7, Y7, Y7").
		Raw("VBROADCASTF128 (DX), Y1").Raw("VPERMILPD $5, Y1, Y4").
		Raw("loop:").Raw("TESTQ CX, CX").Raw("JZ done").
		Raw("VMOVUPD (AX), Y8").Raw("VMOVUPD 32(AX), Y9").
		Raw("VPERM2F128 $32, Y9, Y8, Y5"). // [a0, a1]
		Raw("VPERM2F128 $49, Y9, Y8, Y0"). // [b0, b1]
		Raw("VPERMILPD $0, Y0, Y2").Raw("VPERMILPD $15, Y0, Y3").
		Raw("VMULPD Y1, Y2, Y2").Raw("VMULPD Y4, Y3, Y3").
		Raw("VXORPD Y7, Y3, Y3").Raw("VADDPD Y3, Y2, Y2").
		Raw("VADDPD Y2, Y5, Y8").Raw("VSUBPD Y2, Y5, Y9").
		Raw("VPERM2F128 $32, Y9, Y8, Y0").
		Raw("VPERM2F128 $49, Y9, Y8, Y10").
		Raw("VMOVUPD Y0, (AX)").Raw("VMOVUPD Y10, 32(AX)").
		Raw("ADDQ $64, AX").Raw("SUBQ $4, CX").Raw("JMP loop").
		Raw("done:").Raw("VZEROUPPER").Ret()
	f.Add(b.Func())
}

// genRadix4AVX2 processes two adjacent complex values in each YMM register.
// span must be even. The arithmetic and rounding order match SSE2; shuffles
// stay within 128-bit lanes, so the two butterflies are independent.
func genRadix4AVX2(f *emit.File, name string, inverse bool) {
	sig := amd64.Layout(
		[]string{"a", "n", "span", "w1", "w2", "w3"},
		[]amd64.Type{amd64.Ptr, amd64.Int64, amd64.Int64, amd64.Ptr, amd64.Ptr, amd64.Ptr}, nil, nil,
	)
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("n", "CX").LoadArg("span", "R15")
	// Construct [sign,0,sign,0], then the rotation's [0,sign,0,sign].
	b.Raw("MOVQ $0x8000000000000000, AX").
		Raw("VMOVQ AX, X7").
		Raw("VINSERTF128 $1, X7, Y7, Y7").
		Raw("VPERMILPD $5, Y7, Y6").
		Raw("MOVQ R15, R11").
		Raw("SHLQ $6, R11").
		Raw("SHLQ $4, CX").
		Raw("SHLQ $4, R15").
		Raw("XORQ AX, AX").
		Raw("baseloop:").
		Raw("CMPQ AX, CX").
		Raw("JGE done")
	b.LoadArg("a", "SI").Raw("ADDQ AX, SI").
		Raw("LEAQ (SI)(R15*1), DI").
		Raw("LEAQ (DI)(R15*1), R12").
		Raw("LEAQ (R12)(R15*1), R13")
	b.LoadArg("w1", "R8").LoadArg("w2", "R9").LoadArg("w3", "R10").
		Raw("XORQ R14, R14").
		Raw("kloop:").
		Raw("CMPQ R14, R15").
		Raw("JGE basenext")
	for _, v := range [][3]string{{"DI", "R8", "Y8"}, {"R12", "R9", "Y9"}, {"R13", "R10", "Y10"}} {
		b.Raw("VMOVUPD (" + v[0] + "), Y0").
			Raw("VMOVUPD (" + v[1] + "), Y1").
			Raw("VPERMILPD $0, Y0, Y2").
			Raw("VPERMILPD $15, Y0, Y3").
			Raw("VPERMILPD $5, Y1, Y4").
			Raw("VMULPD Y1, Y2, Y2").
			Raw("VMULPD Y4, Y3, Y3").
			Raw("VXORPD Y7, Y3, Y3").
			Raw("VADDPD Y3, Y2, " + v[2])
	}
	b.Raw("VMOVUPD (SI), Y0").
		Raw("VADDPD Y9, Y0, Y11").
		Raw("VSUBPD Y9, Y0, Y12").
		Raw("VADDPD Y10, Y8, Y13").
		Raw("VSUBPD Y10, Y8, Y14").
		Raw("VPERMILPD $5, Y14, Y15")
	if inverse {
		b.Raw("VXORPD Y7, Y15, Y15")
	} else {
		b.Raw("VXORPD Y6, Y15, Y15")
	}
	b.Raw("VADDPD Y13, Y11, Y0").Raw("VMOVUPD Y0, (SI)").
		Raw("VSUBPD Y13, Y11, Y0").Raw("VMOVUPD Y0, (R12)").
		Raw("VADDPD Y15, Y12, Y0").Raw("VMOVUPD Y0, (DI)").
		Raw("VSUBPD Y15, Y12, Y0").Raw("VMOVUPD Y0, (R13)")
	for _, r := range []string{"SI", "DI", "R12", "R13", "R8", "R9", "R10", "R14"} {
		b.Raw("ADDQ $32, " + r)
	}
	b.Raw("JMP kloop").Raw("basenext:").Raw("ADDQ R11, AX").
		Raw("JMP baseloop").Raw("done:").Raw("VZEROUPPER").Ret()
	f.Add(b.Func())
}

func writeFile(name, content string) {
	if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

// ---------------------------------------------------------------------------
// Stockham pass kernels (the fft package's stockham.go, radices 2/3/4/5/8).
//
// skPass{r}AVX2(cc, ch, tw, k *complex128/float64, ido, l1 int) runs one whole
// Stockham pass: for every block k < l1 and every point i < ido it reads the r
// inputs cc[i+ido·(j+r·k)], applies the size-r butterfly, multiplies output j
// >= 1 by its twiddle and writes ch[i+ido·(k+l1·j)]. Two points (i, i+1) share
// each YMM register; an odd ido finishes with one 128-bit step. ido >= 2.
//
// tw is the pass's twiddle table extended to i = 0: block j-1 (j = 1..r-1)
// holds ido entries. The scalar oracle does NOT multiply at i = 0, and a
// multiply by 1 is not exact on signed zeros, so the first pair of every block
// multiplies both lanes and then takes lane pair 0 back from the unmultiplied
// value (VBLENDPD $3) — bit-identical, not merely equal.
//
// k points at the constants table (32-byte rows, see stockham_amd64.go): row 0
// is the ±i rotation sign mask, which carries the transform direction, so one
// kernel serves both. Every operation mirrors stockham.go's scalar arithmetic
// in the same order with separately rounded VMULPD/VADDPD/VADDSUBPD (no FMA),
// matching the GOAMD64=v1 oracle bit for bit; the rotations are exact lane
// swaps and sign flips.
//
// Registers: AX = input block + i (P), SI = P + 4·S, BX = output + i (O), DI =
// O + 4·OS, R10/R11 = twiddle cursors (T, T + 4·S), CX = S = ido·16, DX = OS =
// l1·ido·16, R12 = 3·S, R13 = 3·OS, R8 = blocks left, R9 = pairs left, R14 =
// constants, R15 = twiddle table base.

const (
	kSign   = 0   // rotation sign mask (direction)
	kHalf   = 32  // 0.5
	kSin120 = 64  // sin(2π/3)
	kC51    = 96  // cos(2π/5)
	kC52    = 128 // cos(4π/5)
	kS51    = 160 // sin(2π/5)
	kS52    = 192 // sin(4π/5)
	kH      = 224 // √2/2
)

type skEmit struct {
	b    *amd64.Builder
	w    string // "Y" (two points) or "X" (the odd tail)
	last int    // radix of a final-pass kernel (ido == 1), 0 for the others
}

func (e skEmit) v(i int) string { return fmt.Sprintf("%s%d", e.w, i) }

func (e skEmit) raw(format string, a ...any) { e.b.Raw(fmt.Sprintf(format, a...)) }

func skIn(j int) string {
	return [...]string{"(AX)", "(AX)(CX*1)", "(AX)(CX*2)", "(AX)(R12*1)", "(SI)", "(SI)(CX*1)", "(SI)(CX*2)", "(SI)(R12*1)"}[j]
}

func skOut(j int) string {
	return [...]string{"(BX)", "(BX)(DX*1)", "(BX)(DX*2)", "(BX)(R13*1)", "(DI)", "(DI)(DX*1)", "(DI)(DX*2)", "(DI)(R13*1)"}[j]
}

// skTw addresses twiddle block j-1 (j >= 1).
func skTw(j int) string {
	return [...]string{"(R10)", "(R10)(CX*1)", "(R10)(CX*2)", "(R10)(R12*1)", "(R11)", "(R11)(CX*1)", "(R11)(CX*2)"}[j-1]
}

func (e skEmit) ld(dst int, addr string) { e.raw("VMOVUPD %s, %s", addr, e.v(dst)) }

// in loads input j. In a final pass (ido == 1) a register holds blocks k and
// k+1 instead of points i and i+1: input j of block k is 16·j bytes into the
// block pair, and of block k+1 16·(r+j).
func (e skEmit) in(dst, j int) {
	if e.last == 0 {
		e.ld(dst, skIn(j))
		return
	}
	e.raw("VMOVUPD %d(AX), X%d", 16*j, dst)
	if e.w == "Y" {
		e.raw("VINSERTF128 $1, %d(AX), Y%d, Y%d", 16*(e.last+j), dst, dst)
	}
}
func (e skEmit) st(src int, addr string)  { e.raw("VMOVUPD %s, %s", e.v(src), addr) }
func (e skEmit) add(dst, a, b int)        { e.raw("VADDPD %s, %s, %s", e.v(b), e.v(a), e.v(dst)) }
func (e skEmit) sub(dst, a, b int)        { e.raw("VSUBPD %s, %s, %s", e.v(b), e.v(a), e.v(dst)) }
func (e skEmit) mulk(dst, a int, off int) { e.raw("VMULPD %d(R14), %s, %s", off, e.v(a), e.v(dst)) }

// rot: dst = rotS(src) — swap re/im, then flip the sign the direction selects.
func (e skEmit) rot(dst, src int) {
	e.raw("VPERMILPD $5, %s, %s", e.v(src), e.v(dst))
	e.raw("VXORPD %d(R14), %s, %s", kSign, e.v(dst), e.v(dst))
}

// twStore multiplies y by twiddle j and stores it to output j. In the first
// pair of a block the i = 0 lane pair keeps the unmultiplied y. A final pass
// has no twiddles: it stores y.
func (e skEmit) twStore(y, j, t1, t2, t3, dst int, first bool) {
	if e.last != 0 {
		e.st(y, skOut(j))
		return
	}
	e.raw("VMOVUPD %s, %s", skTw(j), e.v(t3))
	e.raw("VPERMILPD $0, %s, %s", e.v(y), e.v(t1))
	e.raw("VPERMILPD $15, %s, %s", e.v(y), e.v(t2))
	e.raw("VMULPD %s, %s, %s", e.v(t3), e.v(t1), e.v(t1))
	e.raw("VPERMILPD $5, %s, %s", e.v(t3), e.v(t3))
	e.raw("VMULPD %s, %s, %s", e.v(t3), e.v(t2), e.v(t2))
	e.raw("VADDSUBPD %s, %s, %s", e.v(t2), e.v(t1), e.v(dst))
	if first {
		e.raw("VBLENDPD $3, %s, %s, %s", e.v(y), e.v(dst), e.v(dst))
	}
	e.st(dst, skOut(j))
}

func (e skEmit) body(r int, first bool) {
	switch r {
	case 2:
		e.in(0, 0)
		e.in(1, 1)
		e.add(2, 0, 1)
		e.st(2, skOut(0))
		e.sub(3, 0, 1)
		e.twStore(3, 1, 5, 6, 7, 4, first)
	case 3:
		e.in(0, 0)
		e.in(1, 1)
		e.in(2, 2)
		e.add(3, 1, 2) // t1
		e.sub(4, 1, 2) // t2
		e.add(5, 0, 3)
		e.st(5, skOut(0))
		e.mulk(6, 3, kHalf)
		e.sub(6, 0, 6) // ca = x0 - 0.5·t1
		e.mulk(7, 4, kSin120)
		e.rot(7, 7) // cb
		e.add(1, 6, 7)
		e.sub(2, 6, 7)
		e.twStore(1, 1, 9, 10, 11, 8, first)
		e.twStore(2, 2, 9, 10, 11, 8, first)
	case 4:
		for j := 0; j < 4; j++ {
			e.in(j, j)
		}
		e.add(4, 0, 2) // t2
		e.sub(5, 0, 2) // t1
		e.add(6, 1, 3) // t3
		e.sub(7, 1, 3) // t4
		e.rot(7, 7)
		e.add(0, 4, 6)
		e.st(0, skOut(0))
		e.add(1, 5, 7)
		e.sub(2, 4, 6)
		e.sub(3, 5, 7)
		e.twStore(1, 1, 9, 10, 11, 8, first)
		e.twStore(2, 2, 9, 10, 11, 8, first)
		e.twStore(3, 3, 9, 10, 11, 8, first)
	case 5:
		e.in(0, 0)
		e.in(1, 1)
		e.in(2, 4)
		e.add(3, 1, 2) // t1
		e.sub(4, 1, 2) // t2
		e.in(1, 2)
		e.in(2, 3)
		e.add(5, 1, 2) // t3
		e.sub(6, 1, 2) // t4
		e.add(7, 0, 3)
		e.add(7, 7, 5)
		e.st(7, skOut(0))
		e.mulk(8, 3, kC51)
		e.mulk(9, 5, kC52)
		e.add(8, 8, 9)
		e.add(8, 0, 8) // r1
		e.mulk(9, 3, kC52)
		e.mulk(10, 5, kC51)
		e.add(9, 9, 10)
		e.add(9, 0, 9) // r2
		e.mulk(10, 4, kS51)
		e.mulk(11, 6, kS52)
		e.add(10, 10, 11)
		e.rot(10, 10) // i1
		e.mulk(11, 4, kS52)
		e.mulk(12, 6, kS51)
		e.sub(11, 11, 12)
		e.rot(11, 11)   // i2
		e.add(1, 8, 10) // y1
		e.sub(4, 8, 10) // y4
		e.add(2, 9, 11) // y2
		e.sub(3, 9, 11) // y3
		e.twStore(1, 1, 12, 13, 14, 15, first)
		e.twStore(2, 2, 12, 13, 14, 15, first)
		e.twStore(3, 3, 12, 13, 14, 15, first)
		e.twStore(4, 4, 12, 13, 14, 15, first)
	case 8:
		e.in(1, 1)
		e.in(5, 5)
		e.add(8, 1, 5) // a1
		e.sub(9, 1, 5) // a5
		e.in(3, 3)
		e.in(7, 7)
		e.add(10, 3, 7) // a3
		e.sub(11, 3, 7) // a7
		e.add(12, 8, 10)
		e.sub(10, 8, 10) // a1, a3 = a1+a3, a1-a3
		e.rot(10, 10)
		e.rot(11, 11)
		e.add(8, 9, 11)
		e.sub(9, 9, 11) // a5, a7 = a5+a7, a5-a7
		e.rot(11, 8)
		e.add(8, 8, 11)
		e.mulk(8, 8, kH) // a5 = h·(a5 + rotS(a5))
		e.rot(11, 9)
		e.sub(9, 11, 9)
		e.mulk(9, 9, kH) // a7 = h·(rotS(a7) - a7)
		e.in(0, 0)
		e.in(4, 4)
		e.add(1, 0, 4) // a0
		e.sub(2, 0, 4) // a4
		e.in(0, 2)
		e.in(4, 6)
		e.add(3, 0, 4) // a2
		e.sub(5, 0, 4) // a6
		e.add(0, 1, 3)
		e.sub(1, 1, 3) // a0, a2 = a0+a2, a0-a2
		e.rot(5, 5)
		e.add(3, 2, 5)
		e.sub(2, 2, 5) // a4, a6 = a4+a6, a4-a6
		// a0=0 a2=1 a4=3 a6=2 a1=12 a3=10 a5=8 a7=9
		e.add(4, 0, 12)
		e.st(4, skOut(0))
		e.sub(0, 0, 12)  // y4
		e.add(12, 1, 10) // y2
		e.sub(1, 1, 10)  // y6
		e.add(10, 3, 8)  // y1
		e.sub(3, 3, 8)   // y5
		e.add(8, 2, 9)   // y3
		e.sub(2, 2, 9)   // y7
		for _, yj := range [][2]int{{10, 1}, {12, 2}, {8, 3}, {0, 4}, {3, 5}, {1, 6}, {2, 7}} {
			e.twStore(yj[0], yj[1], 13, 14, 15, 4, first)
		}
	}
}

func (e skEmit) advance(bytes int) {
	for _, r := range []string{"AX", "SI", "BX", "DI", "R10", "R11"} {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

func genStockhamAVX2(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skPass%dAVX2", r), sig, 0)
	y, x := skEmit{b, "Y", 0}, skEmit{b, "X", 0}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("ido", "CX").LoadArg("l1", "R8")
	b.Raw("MOVQ CX, DX").
		Raw("IMULQ R8, DX").
		Raw("SHLQ $4, CX"). // S
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("kloop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("MOVQ R15, R10").
		Raw("LEAQ (R15)(CX*4), R11")
	y.body(r, true)
	y.advance(32)
	b.Raw("MOVQ CX, R9").
		Raw("SHRQ $5, R9"). // pairs = ido/2
		Raw("DECQ R9").
		Raw("iloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ tail")
	y.body(r, false)
	y.advance(32)
	b.Raw("DECQ R9").
		Raw("JMP iloop").
		Raw("tail:").
		Raw("TESTQ $16, CX"). // ido odd
		Raw("JZ knext")
	x.body(r, false)
	x.advance(16)
	b.Raw("knext:")
	// P advanced by S over the block; the next block starts r·S after this one.
	switch r {
	case 2:
		b.Raw("ADDQ CX, AX")
	case 3:
		b.Raw("LEAQ (AX)(CX*2), AX")
	case 4:
		b.Raw("ADDQ R12, AX")
	case 5:
		b.Raw("LEAQ (AX)(CX*4), AX")
	case 8:
		b.Raw("LEAQ (AX)(CX*4), AX").Raw("ADDQ R12, AX")
	}
	// O advanced by S, which is exactly the next block's output start.
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genStockhamLastAVX2 emits skLast{r}AVX2(cc, ch, k, l1): the final Stockham
// pass (ido == 1), where every block is a single point and there are no
// twiddles. It vectorizes across blocks instead of points — blocks k and k+1
// share each YMM register, their inputs gathered from the two adjacent r-point
// blocks, their outputs landing contiguous in each of the r output runs — and
// an odd l1 finishes with one 128-bit step. The butterflies are the ones the
// other passes use, so the result is again bit-identical to stockham.go's
// pass{r}last.
func genStockhamLastAVX2(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skLast%dAVX2", r), sig, 0)
	y, x := skEmit{b, "Y", r}, skEmit{b, "X", r}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("l1", "DX")
	b.Raw("MOVQ DX, R9").
		Raw("SHRQ $1, R9"). // block pairs
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("loop:").
		Raw("TESTQ R9, R9").
		Raw("JZ tail")
	y.body(r, false)
	b.Raw(fmt.Sprintf("ADDQ $%d, AX", 32*r)).
		Raw("ADDQ $32, BX").
		Raw("ADDQ $32, DI").
		Raw("DECQ R9").
		Raw("JMP loop").
		Raw("tail:").
		Raw("TESTQ $16, DX"). // l1 odd
		Raw("JZ done")
	x.body(r, false)
	b.Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

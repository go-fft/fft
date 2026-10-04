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
	b.Raw("MOVUPD (%s), X0", xPtr). // a = [ar,ai]
					Raw("MOVUPD (%s), X1", yPtr). // b = [br,bi]
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
		b.Raw("MOVAPS X2, %s", dst)
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
	// AVX-512 only for the radices of a power-of-two transform, the only
	// transforms it measured faster on (see stockham_amd64.go).
	for _, r := range []int{2, 4, 8} {
		genStockhamAVX512(fb, r)
		genStockhamLastAVX512(fb, r)
	}
	genUntangleAVX2(fb)
	genRetangleAVX2(fb)
	writeFile("butterfly_amd64.s", fb.String())

	// The gate in front of those AVX2 kernels. It used to be 24 hand-written
	// lines in cpu_amd64.s; go-asmgen v0.9.0 emits the same sequence, and the
	// OS half -- OSXSAVE, then XGETBV's XMM and YMM bits in XCR0, before the
	// feature bit -- is the part a hand-rolled probe drops. A CPU can report
	// AVX2 on a kernel that does not save YMM state, and then the upper lanes
	// go at the next context switch.
	fp := emit.NewFile("amd64")
	fp.Add(amd64.FeatureProbe("supportsAVX2", amd64.AVX2))
	// AVX-512 also needs the OS to save opmask and ZMM state (XCR0 0xE6);
	// go-asmgen v0.10.0 checks it. On macOS it answers false (Darwin enables
	// ZMM state lazily) and the AVX2 kernels run.
	fp.Add(amd64.FeatureProbe("supportsAVX512F", amd64.AVX512F))
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
		b.Raw("VMOVUPD (%s), Y0", v[0]).
			Raw("VMOVUPD (%s), Y1", v[1]).
			Raw("VPERMILPD $0, Y0, Y2").
			Raw("VPERMILPD $15, Y0, Y3").
			Raw("VPERMILPD $5, Y1, Y4").
			Raw("VMULPD Y1, Y2, Y2").
			Raw("VMULPD Y4, Y3, Y3").
			Raw("VXORPD Y7, Y3, Y3").
			Raw("VADDPD Y3, Y2, %s", v[2])
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
		b.Raw("ADDQ $32, %s", r)
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
	// kAddSub, in the AVX-512 table only: [-0, 0] repeated, the real-lane
	// sign flip that stands in for VADDSUBPD, which has no 512-bit form.
	kAddSub = 256
)

type skEmit struct {
	b    *amd64.Builder
	w    string // "Z" (four points), "Y" (two) or "X" (one, the odd tail)
	last int    // radix of a final-pass kernel (ido == 1), 0 for the others
	wide bool   // inside an AVX-512 kernel: 64-byte constant rows
}

// k scales a constants-table offset (given for 32-byte rows) to the table the
// kernel reads: the AVX-512 table's rows are 64 bytes.
func (e skEmit) k(off int) int {
	if e.wide {
		return 2 * off
	}
	return off
}

// perm emits a lane permute within each complex: "lo" broadcasts the real
// part, "hi" the imaginary part, "swap" exchanges them. The immediates differ
// with the register width (one bit per lane).
func (e skEmit) perm(kind string, src, dst int) {
	imm := map[string]string{"lo": "$0", "hi": "$15", "swap": "$5"}[kind]
	if e.w == "Z" {
		imm = map[string]string{"lo": "$0x00", "hi": "$0xFF", "swap": "$0x55"}[kind]
	}
	e.raw("VPERMILPD %s, %s, %s", imm, e.v(src), e.v(dst))
}

// xork flips signs by a constants row: VPXORQ for ZMM (VXORPD on ZMM would
// need AVX512DQ), VXORPD otherwise.
func (e skEmit) xork(off, r int) {
	op := "VXORPD"
	if e.w == "Z" {
		op = "VPXORQ"
	}
	e.raw("%s %d(R14), %s, %s", op, e.k(off), e.v(r), e.v(r))
}

func (e skEmit) v(i int) string { return fmt.Sprintf("%s%d", e.w, i) }

func (e skEmit) raw(format string, a ...any) { e.b.Raw(format, a...) }

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
	switch e.w {
	case "Y":
		e.raw("VINSERTF128 $1, %d(AX), Y%d, Y%d", 16*(e.last+j), dst, dst)
	case "Z":
		// Four blocks: VINSERTF32X4 is AVX512F (its 64x2 twin needs DQ) and
		// moves the same 128 bits.
		for q := 1; q < 4; q++ {
			e.raw("VINSERTF32X4 $%d, %d(AX), Z%d, Z%d", q, 16*(q*e.last+j), dst, dst)
		}
	}
}
func (e skEmit) st(src int, addr string) { e.raw("VMOVUPD %s, %s", e.v(src), addr) }
func (e skEmit) add(dst, a, b int)       { e.raw("VADDPD %s, %s, %s", e.v(b), e.v(a), e.v(dst)) }
func (e skEmit) sub(dst, a, b int)       { e.raw("VSUBPD %s, %s, %s", e.v(b), e.v(a), e.v(dst)) }
func (e skEmit) mulk(dst, a int, off int) {
	e.raw("VMULPD %d(R14), %s, %s", e.k(off), e.v(a), e.v(dst))
}

// rot: dst = rotS(src) — swap re/im, then flip the sign the direction selects.
func (e skEmit) rot(dst, src int) {
	e.perm("swap", src, dst)
	e.xork(kSign, dst)
}

// twStore multiplies y by twiddle j and stores it to output j. In the first
// pair of a block the i = 0 lane pair keeps the unmultiplied y. A final pass
// has no twiddles: it stores y.
//
// The twiddle's halves are taken straight from memory: VMOVDDUP gives
// (wr, wr) per point as a plain load, and VPERMILPD with a memory operand
// gives (wi, wi), so the multiply takes two shuffles (that one and swapping
// y) where broadcasting y's halves and swapping w took three, all on the one
// shuffle port Intel cores have. It computes (yr·wr − yi·wi, yi·wr + yr·wi):
// the same products as the scalar oracle's (yr·wr − yi·wi, yr·wi + yi·wr),
// its imaginary sum taken in the other order, which IEEE addition makes
// exact. Loading (wi, wi) with a second VMOVDDUP at w+8 saves the other
// shuffle but straddles a cache line every other pair: it measured 8% slower
// at 2048 and 4096 on Zen 3, where this form is within 2% of the old one or
// faster at every size (2026-10-04).
func (e skEmit) twStore(y, j, t1, t2, t3, dst int, first bool) {
	if e.last != 0 {
		e.st(y, skOut(j))
		return
	}
	hi := map[string]string{"X": "$3", "Y": "$15", "Z": "$0xFF"}[e.w]
	e.raw("VMOVDDUP %s, %s", skTw(j), e.v(t1))
	e.raw("VPERMILPD %s, %s, %s", hi, skTw(j), e.v(t2))
	e.perm("swap", y, t3)
	e.raw("VMULPD %s, %s, %s", e.v(y), e.v(t1), e.v(t1))
	e.raw("VMULPD %s, %s, %s", e.v(t3), e.v(t2), e.v(t2))
	if e.w == "Z" {
		// No 512-bit VADDSUBPD: flip the real lanes' sign, then add.
		// x + (-y) is x - y exactly, so the rounding is VADDSUBPD's.
		e.xork(kAddSub, t2)
		e.raw("VADDPD %s, %s, %s", e.v(t2), e.v(t1), e.v(dst))
	} else {
		e.raw("VADDSUBPD %s, %s, %s", e.v(t2), e.v(t1), e.v(dst))
	}
	if first {
		if e.w == "Z" {
			// K1 = lanes 0–1: point i = 0 keeps the unmultiplied y.
			e.raw("VBLENDMPD %s, %s, K1, %s", e.v(y), e.v(dst), e.v(dst))
		} else {
			e.raw("VBLENDPD $3, %s, %s, %s", e.v(y), e.v(dst), e.v(dst))
		}
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
	y, x := skEmit{b, "Y", 0, false}, skEmit{b, "X", 0, false}
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
	y, x := skEmit{b, "Y", r, false}, skEmit{b, "X", r, false}
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
	b.Raw("ADDQ $%d, AX", 32*r).
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

// genUntangleAVX2 emits untangleAVX2(dst, z, tw *complex128, k *skConst, m,
// pairs int): the real-FFT untangle (the fft package's rfftUntangle) for bins
// k = 1 .. 2·pairs, two consecutive k per YMM register. Bin k reads Z[k] and
// Z[m-k]; for k and k+1 those are one forward load of Z[k..k+1] and one load of
// Z[m-k-1..m-k] with its 128-bit halves swapped, and the two mirrored outputs
// dst[m-k], dst[m-k-1] are stored the same way. Per bin, with Z[k] = a and
// Z[m-k] = b, s = a + b and d = a - b:
//
//	xe = [s.re, d.im]·0.5          (a blend, then the multiply)
//	xo = [s.im, -d.re]·0.5         (the other blend, a swap, a sign flip)
//	t  = W^k · xo                  (the complex product, VADDSUBPD)
//	dst[k] = xe + t;  dst[m-k] = conj(xe - t)
//
// which is rfftUntangle's arithmetic operation for operation (the scalar code
// spells the same values out on real and imaginary parts), separately rounded
// with no FMA: bit-identical at GOAMD64=v1. k's row 0 (the forward rotation
// mask, [0, -0, 0, -0]) is exactly the imaginary-lane sign flip, row 1 is 0.5.
func genUntangleAVX2(f *emit.File) {
	sig := amd64.Layout(
		[]string{"dst", "z", "tw", "k", "m", "pairs"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("untangleAVX2", sig, 0)
	b.LoadArg("dst", "DI").LoadArg("z", "SI").LoadArg("tw", "CX").LoadArg("k", "R14").
		LoadArg("m", "DX").LoadArg("pairs", "R8")
	b.Raw("SHLQ $4, DX"). // m·16
				Raw("LEAQ 16(SI), AX").         // &Z[1]
				Raw("LEAQ -32(SI)(DX*1), BX").  // &Z[m-2] = pair (m-2, m-1)
				Raw("LEAQ 16(DI), R9").         // &dst[1]
				Raw("LEAQ -32(DI)(DX*1), R10"). // &dst[m-2]
				Raw("ADDQ $16, CX").            // &tw[1]
				Raw("loop:").
				Raw("TESTQ R8, R8").
				Raw("JZ done").
				Raw("VMOVUPD (AX), Y0").             // a = Z[k], Z[k+1]
				Raw("VMOVUPD (BX), Y1").             // Z[m-k-1], Z[m-k]
				Raw("VPERM2F128 $1, Y1, Y1, Y1").    // b = Z[m-k], Z[m-k-1]
				Raw("VADDPD Y1, Y0, Y2").            // s
				Raw("VSUBPD Y1, Y0, Y3").            // d
				Raw("VBLENDPD $10, Y3, Y2, Y4").     // [s.re, d.im]
				Raw("VMULPD 32(R14), Y4, Y4").       // xe
				Raw("VBLENDPD $5, Y3, Y2, Y5").      // [d.re, s.im]
				Raw("VPERMILPD $5, Y5, Y5").         // [s.im, d.re]
				Raw("VXORPD 0(R14), Y5, Y5").        // [s.im, -d.re]
				Raw("VMULPD 32(R14), Y5, Y5").       // xo
				Raw("VMOVUPD (CX), Y6").             // w
				Raw("VPERMILPD $0, Y6, Y7").         // [wr, wr]
				Raw("VPERMILPD $15, Y6, Y8").        // [wi, wi]
				Raw("VPERMILPD $5, Y5, Y9").         // [xo.im, xo.re]
				Raw("VMULPD Y5, Y7, Y7").            // [wr·xo.re, wr·xo.im]
				Raw("VMULPD Y9, Y8, Y8").            // [wi·xo.im, wi·xo.re]
				Raw("VADDSUBPD Y8, Y7, Y7").         // t = w·xo
				Raw("VADDPD Y7, Y4, Y10").           // dst[k], dst[k+1]
				Raw("VSUBPD Y7, Y4, Y11").           // xe - t
				Raw("VXORPD 0(R14), Y11, Y11").      // conj: dst[m-k], dst[m-k-1]
				Raw("VPERM2F128 $1, Y11, Y11, Y11"). // dst[m-k-1], dst[m-k]
				Raw("VMOVUPD Y10, (R9)").
				Raw("VMOVUPD Y11, (R10)").
				Raw("ADDQ $32, AX").
				Raw("SUBQ $32, BX").
				Raw("ADDQ $32, R9").
				Raw("SUBQ $32, R10").
				Raw("ADDQ $32, CX").
				Raw("DECQ R8").
				Raw("JMP loop").
				Raw("done:").
				Raw("VZEROUPPER").
				Ret()
	f.Add(b.Func())
}

// genRetangleAVX2 emits retangleAVX2(z, x, tw *complex128, k *skConst, h
// *[4]float64, m, pairs int): the inverse of the untangle (the fft package's
// irfftRetangle) for k = 1 .. 2·pairs, two consecutive k per YMM register,
// with the same forward/mirrored load and store pairing. Per bin, with X[k] =
// a, X[m-k] = b, s = a + b, d = a - b and h = 0.5·scale:
//
//	xe = [s.re, d.im]·h;  dd = [d.re, s.im]·h
//	xo = conj(W^k)·dd     (conj: an exact sign flip of W's imaginary lane)
//	Z[k] = xe + i·xo;  Z[m-k] = conj(xe - i·xo)
//
// operation for operation retangle's arithmetic, separately rounded, no FMA.
// k's row 0 ([0,-0,0,-0]) flips imaginary lanes; skInv's row 0 would flip
// real ones, so i·xo is a lane swap then a flip of the real lanes, done with
// the imaginary mask after the swap.
func genRetangleAVX2(f *emit.File) {
	sig := amd64.Layout(
		[]string{"z", "x", "tw", "k", "h", "m", "pairs"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("retangleAVX2", sig, 0)
	b.LoadArg("z", "DI").LoadArg("x", "SI").LoadArg("tw", "CX").LoadArg("k", "R14").
		LoadArg("h", "R15").LoadArg("m", "DX").LoadArg("pairs", "R8")
	b.Raw("SHLQ $4, DX").
		Raw("LEAQ 16(SI), AX").         // &X[1]
		Raw("LEAQ -32(SI)(DX*1), BX").  // &X[m-2]
		Raw("LEAQ 16(DI), R9").         // &Z[1]
		Raw("LEAQ -32(DI)(DX*1), R10"). // &Z[m-2]
		Raw("ADDQ $16, CX").            // &tw[1]
		Raw("VMOVUPD (R15), Y12").      // h
		Raw("VMOVUPD 0(R14), Y13").     // imaginary-lane sign mask
		Raw("loop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("VMOVUPD (AX), Y0"). // a = X[k], X[k+1]
		Raw("VMOVUPD (BX), Y1").
		Raw("VPERM2F128 $1, Y1, Y1, Y1"). // b = X[m-k], X[m-k-1]
		Raw("VADDPD Y1, Y0, Y2").         // s
		Raw("VSUBPD Y1, Y0, Y3").         // d
		Raw("VBLENDPD $10, Y3, Y2, Y4").  // [s.re, d.im]
		Raw("VMULPD Y12, Y4, Y4").        // xe
		Raw("VBLENDPD $5, Y3, Y2, Y5").   // [d.re, s.im]
		Raw("VMULPD Y12, Y5, Y5").        // dd = [dr, di]
		Raw("VMOVUPD (CX), Y6").
		Raw("VXORPD Y13, Y6, Y6").    // conj(w) = [wr, -wi]
		Raw("VPERMILPD $0, Y6, Y7").  // [wr, wr]
		Raw("VPERMILPD $15, Y6, Y8"). // [-wi, -wi]
		Raw("VPERMILPD $5, Y5, Y9").  // [di, dr]
		Raw("VMULPD Y5, Y7, Y7").     // [wr·dr, wr·di]
		Raw("VMULPD Y9, Y8, Y8").     // [-wi·di, -wi·dr]
		Raw("VADDSUBPD Y8, Y7, Y7").  // xo = [wr·dr + wi·di, wr·di - wi·dr]
		Raw("VPERMILPD $5, Y7, Y8").  // [xoi, xor]
		Raw("VXORPD Y13, Y8, Y8").    // [xoi, -xor]
		Raw("VSUBPD Y8, Y4, Y10").    // xe - [xoi, -xor] = [xer - xoi, xei + xor] = Z[k]
		Raw("VADDPD Y8, Y4, Y11").    // [xer + xoi, xei - xor]
		Raw("VXORPD Y13, Y11, Y11").  // Z[m-k] = [xer + xoi, -(xei - xor)]
		Raw("VPERM2F128 $1, Y11, Y11, Y11").
		Raw("VMOVUPD Y10, (R9)").
		Raw("VMOVUPD Y11, (R10)").
		Raw("ADDQ $32, AX").
		Raw("SUBQ $32, BX").
		Raw("ADDQ $32, R9").
		Raw("SUBQ $32, R10").
		Raw("ADDQ $32, CX").
		Raw("DECQ R8").
		Raw("JMP loop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// knext emits the advance from one Stockham block to the next: P has moved
// by S over the block, and the next block starts r·S after this one.
func skNextBlock(b *amd64.Builder, r int) {
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
}

// genStockhamAVX512 emits skPass{r}AVX512: genStockhamAVX2's pass with four
// points per ZMM register. It needs ido >= 4; ido mod 4 is finished with one
// 256-bit and/or one 128-bit step, the same bodies the AVX2 kernel runs. The
// constants table has 64-byte rows (skConst512). Every substitution for an
// instruction AVX-512 lacks at this width is exact: VADDSUBPD becomes a sign
// flip and an add, VXORPD becomes VPXORQ, the i = 0 blend uses opmask K1.
func genStockhamAVX512(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skPass%dAVX512", r), sig, 0)
	z, y, x := skEmit{b, "Z", 0, true}, skEmit{b, "Y", 0, true}, skEmit{b, "X", 0, true}
	b.Raw("MOVQ $3, R9").Raw("KMOVB R9, K1") // lanes 0–1: the i = 0 point
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("ido", "CX").LoadArg("l1", "R8")
	b.Raw("MOVQ CX, DX").
		Raw("IMULQ R8, DX").
		Raw("SHLQ $4, CX").
		Raw("SHLQ $4, DX").
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("kloop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("MOVQ R15, R10").
		Raw("LEAQ (R15)(CX*4), R11")
	z.body(r, true)
	z.advance(64)
	b.Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9"). // quads = ido/4
		Raw("DECQ R9").
		Raw("iloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	z.body(r, false)
	z.advance(64)
	b.Raw("DECQ R9").
		Raw("JMP iloop").
		Raw("pair:").
		Raw("TESTQ $32, CX"). // ido & 2
		Raw("JZ single")
	y.body(r, false)
	y.advance(32)
	b.Raw("single:").
		Raw("TESTQ $16, CX"). // ido & 1
		Raw("JZ knext")
	x.body(r, false)
	x.advance(16)
	b.Raw("knext:")
	skNextBlock(b, r)
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genStockhamLastAVX512 emits skLast{r}AVX512: the final pass (ido == 1) with
// four blocks per ZMM register (VINSERTF32X4 gathers blocks k..k+3); l1 mod 4
// is finished with one 256-bit and/or one 128-bit step.
func genStockhamLastAVX512(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skLast%dAVX512", r), sig, 0)
	z, y, x := skEmit{b, "Z", r, true}, skEmit{b, "Y", r, true}, skEmit{b, "X", r, true}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("l1", "DX")
	b.Raw("MOVQ DX, R9").
		Raw("SHRQ $2, R9"). // block quads
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("loop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	z.body(r, false)
	b.Raw("ADDQ $%d, AX", 64*r).
		Raw("ADDQ $64, BX").
		Raw("ADDQ $64, DI").
		Raw("DECQ R9").
		Raw("JMP loop").
		Raw("pair:").
		Raw("TESTQ $32, DX"). // l1 & 2
		Raw("JZ single")
	y.body(r, false)
	b.Raw("ADDQ $%d, AX", 32*r).
		Raw("ADDQ $32, BX").
		Raw("ADDQ $32, DI").
		Raw("single:").
		Raw("TESTQ $16, DX"). // l1 & 1
		Raw("JZ done")
	x.body(r, false)
	b.Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

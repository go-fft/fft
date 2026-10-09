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
	"strings"

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
	for _, r := range []int{2, 3, 4, 5, 8} {
		genStockhamBatchAVX2(fb, r)
	}
	// Radix 16 at 256 bits only: the fft package never factors with radix 16
	// where the AVX-512 kernels run (see radix16TableAMD64).
	genStockham16(fb)
	genStockham16Last(fb)
	// Radix 12 (Round 24), for composites, which never run the AVX-512 kernels.
	genComp12(fb)
	genComp12Last(fb)
	// The final pass of the blocked schedule for large powers of two
	// (cascadeBlocked in the fft package): the radices that end a power of two.
	for _, r := range []int{2, 4, 8} {
		genStockhamLastRun(fb, r, false)
		genStockhamLastRun(fb, r, true)
	}
	// The two passes it runs over the whole array chunk by chunk
	// (cascadePair): the radices that open a large power of two.
	for _, r := range []int{4, 8} {
		genStockhamStrided(fb, r, false)
		genStockhamStrided(fb, r, true)
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
	// AVX-512 also needs the OS to save opmask and ZMM state (XCR0 0xE6);
	// go-asmgen v0.10.0 checks it. On macOS it answers false (Darwin enables
	// ZMM state lazily) and the AVX2 kernels run.
	fp.Add(amd64.FeatureProbe("supportsAVX512F", amd64.AVX512F))
	// The vendor, for a tuning choice measured to differ between Intel and
	// AMD (the radix rule, see the fft package's route_amd64.go); go-asmgen
	// v0.12.0. Never for correctness: the kernels' results do not depend on it.
	fp.Add(amd64.VendorProbe("isGenuineIntel", "GenuineIntel"))
	writeFile("cpu_amd64.s", fp.String())

	genF32StockhamFile()
	genSplitFile()
	genF32RealFile()
	genComp2File()
	genIntelSplit512File()
	genSmallUntangleFile()
	genZnTwoPassFile()
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
// tw is the pass's twiddle table extended to i = 0, laid out group by group
// (see StockhamTwiddles and skEmit.tw): the points are walked in groups of
// four (two pairs), then one pair and one single point for ido mod 4, and each
// group's r-1 twiddle runs are stored together. The scalar oracle does NOT multiply at i = 0, and a
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
// O + 4·OS, R10 = twiddle group cursor, CX = S = ido·16, DX = OS = l1·ido·16,
// R12 = 3·S, R13 = 3·OS, R8 = blocks left, R9 = groups left, R14 = constants,
// R15 = twiddle table base.

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
	// The radix-16 butterfly's internal twiddles W16^e = cos θ + s·i·sin θ
	// (θ = 2π·e/16; kernels.skConst rows 9..16): the cosines, in every
	// lane, and the rows [−ss, ss, ...] with ss = s·sin θ, which carry the
	// direction. kH is the cosine of e = 2.
	kR16C1   = 288 // cos(π/8): e = 1
	kR16S1   = 320 // sin(π/8) = cos(3π/8): e = 3
	kR16NH   = 352 // −√2/2 = cos(3π/4): e = 6
	kR16NC1  = 384 // −cos(π/8) = cos(9π/8): e = 9
	kR16SS1  = 416 // ss = s·sin(π/8): e = 1
	kR16SSH  = 448 // ss = s·√2/2: e = 2 and 6
	kR16SSC1 = 480 // ss = s·cos(π/8) = s·sin(3π/8): e = 3
	kR16SSN1 = 512 // ss = −s·sin(π/8) = s·sin(9π/8): e = 9
)

type skEmit struct {
	b    *amd64.Builder
	w    string // "Z" (four points), "Y" (two) or "X" (one, the odd tail)
	last int    // radix of a final-pass kernel (ido == 1), 0 for the others
	wide bool   // inside an AVX-512 kernel: 64-byte constant rows
	// g is the size, in points, of the twiddle group the body reads (see
	// StockhamTwiddles): 4, 2 or 1. half is 1 for the second pair of a
	// four-point group walked as two pairs, 0 otherwise.
	g, half int
	// notw stores every output without a twiddle (the i = 0 point of a
	// batched pass); bcast multiplies by one twiddle per output broadcast to
	// every lane (the other points of a batched pass), read from (j-1)·16(R10).
	notw, bcast bool
	// dup reads the radix-16 twiddle table, whose twiddles are stored
	// pre-duplicated: (wr, wr) and (wi, wi) (see radix16Twiddles).
	dup bool
	// addr, when set, addresses input (out false) or output stream j in
	// place of skIn and skOut: the radix-10/15/20 kernels (Round 26) have
	// more streams than those reach.
	addr func(out bool, j int) string
}

// inAddr and outAddr address input and output stream j.
func (e skEmit) inAddr(j int) string {
	if e.addr != nil {
		return e.addr(false, j)
	}
	return skIn(j)
}

func (e skEmit) outAddr(j int) string {
	if e.addr != nil {
		return e.addr(true, j)
	}
	return skOut(j)
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

// skIn and skOut address input and output stream j. Streams 8..15 exist only
// in the radix-16 kernels, whose third and fourth base registers (R8, R9 in,
// R11, R15 out) the other kernels use for something else.
func skIn(j int) string {
	return [...]string{"(AX)", "(AX)(CX*1)", "(AX)(CX*2)", "(AX)(R12*1)", "(SI)", "(SI)(CX*1)", "(SI)(CX*2)", "(SI)(R12*1)",
		"(R8)", "(R8)(CX*1)", "(R8)(CX*2)", "(R8)(R12*1)", "(R9)", "(R9)(CX*1)", "(R9)(CX*2)", "(R9)(R12*1)"}[j]
}

func skOut(j int) string {
	return [...]string{"(BX)", "(BX)(DX*1)", "(BX)(DX*2)", "(BX)(R13*1)", "(DI)", "(DI)(DX*1)", "(DI)(DX*2)", "(DI)(R13*1)",
		"(R11)", "(R11)(DX*1)", "(R11)(DX*2)", "(R11)(R13*1)", "(R15)", "(R15)(DX*1)", "(R15)(DX*2)", "(R15)(R13*1)"}[j]
}

// tw addresses twiddle j (j >= 1) of the points the body runs on. The table
// holds the twiddles group by group, in the order the kernel walks the points:
// a group of g points stores its r-1 twiddle runs one after the other, g
// entries each, so twiddle j sits (j-1)·16·g bytes into the group (plus 32 for
// the second pair of a four-point group). R10 points at the current group. A
// pass then reads its twiddles as one sequential stream instead of r-1 runs
// ido entries apart.
func (e skEmit) tw(j int) string {
	return fmt.Sprintf("%d(R10)", (j-1)*16*e.g+32*e.half)
}

// twNext moves R10 past the current group.
func (e skEmit) twNext(r int) {
	if e.dup {
		e.raw("ADDQ $%d, R10", (r-1)*32*e.g)
		return
	}
	e.raw("ADDQ $%d, R10", (r-1)*16*e.g)
}

// twDup addresses the pre-duplicated halves of twiddle j: a group of g points
// stores, per j, the g pairs (wr, wr) then the g pairs (wi, wi).
func (e skEmit) twDup(j int) (wr, wi string) {
	at := (j-1)*32*e.g + 32*e.half
	return fmt.Sprintf("%d(R10)", at), fmt.Sprintf("%d(R10)", at+16*e.g)
}

func (e skEmit) ld(dst int, addr string) { e.raw("VMOVUPD %s, %s", addr, e.v(dst)) }

// in loads input j. In a final pass (ido == 1) a register holds blocks k and
// k+1 instead of points i and i+1: input j of block k is 16·j bytes into the
// block pair, and of block k+1 16·(r+j).
func (e skEmit) in(dst, j int) {
	if e.last == 0 {
		e.ld(dst, e.inAddr(j))
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
	if e.last != 0 || e.notw {
		e.st(y, e.outAddr(j))
		return
	}
	if e.bcast {
		// One twiddle for the whole register: its halves broadcast straight
		// from memory, VBROADCASTSD (VMOVDDUP at 128 bits), which are plain
		// loads, so the product costs one shuffle, the swap of y.
		ld := "VBROADCASTSD"
		if e.w == "X" {
			ld = "VMOVDDUP"
		}
		e.raw("%s %d(R10), %s", ld, (j-1)*16, e.v(t1))
		e.raw("%s %d(R10), %s", ld, (j-1)*16+8, e.v(t2))
		e.perm("swap", y, t3)
		e.raw("VMULPD %s, %s, %s", e.v(y), e.v(t1), e.v(t1))
		e.raw("VMULPD %s, %s, %s", e.v(t3), e.v(t2), e.v(t2))
		e.raw("VADDSUBPD %s, %s, %s", e.v(t2), e.v(t1), e.v(dst))
		e.st(dst, e.outAddr(j))
		return
	}
	if e.dup {
		// The halves come pre-duplicated: each product takes its twiddle
		// half as a memory operand, and swapping y is the only shuffle.
		wr, wi := e.twDup(j)
		e.perm("swap", y, t3)
		e.raw("VMULPD %s, %s, %s", wr, e.v(y), e.v(t1))
		e.raw("VMULPD %s, %s, %s", wi, e.v(t3), e.v(t2))
	} else {
		hi := map[string]string{"X": "$3", "Y": "$15", "Z": "$0xFF"}[e.w]
		e.raw("VMOVDDUP %s, %s", e.tw(j), e.v(t1))
		e.raw("VPERMILPD %s, %s, %s", hi, e.tw(j), e.v(t2))
		e.perm("swap", y, t3)
		e.raw("VMULPD %s, %s, %s", e.v(y), e.v(t1), e.v(t1))
		e.raw("VMULPD %s, %s, %s", e.v(t3), e.v(t2), e.v(t2))
	}
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
	e.st(dst, e.outAddr(j))
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
	for _, r := range []string{"AX", "SI", "BX", "DI"} {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

func genStockhamAVX2(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skPass%dAVX2", r), sig, 0)
	// A four-point twiddle group is walked as two pairs (q0, q1); then come a
	// two-point group (p) and a one-point group (x) for ido mod 4.
	q0 := skEmit{b: b, w: "Y", g: 4}
	q1 := skEmit{b: b, w: "Y", g: 4, half: 1}
	p := skEmit{b: b, w: "Y", g: 2}
	x := skEmit{b: b, w: "X", g: 1}
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
		Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9"). // four-point groups = ido/4
		Raw("TESTQ R9, R9").
		Raw("JZ firstpair")
	q0.body(r, true)
	q0.advance(32)
	q1.body(r, false)
	q1.advance(32)
	q0.twNext(r)
	b.Raw("DECQ R9").
		Raw("qloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	q0.body(r, false)
	q0.advance(32)
	q1.body(r, false)
	q1.advance(32)
	q0.twNext(r)
	b.Raw("DECQ R9").
		Raw("JMP qloop").
		Raw("pair:").
		Raw("TESTQ $32, CX"). // ido & 2
		Raw("JZ single")
	p.body(r, false)
	p.advance(32)
	p.twNext(r)
	b.Raw("JMP single").
		Raw("firstpair:") // ido is 2 or 3: the first pair is the two-point group
	p.body(r, true)
	p.advance(32)
	p.twNext(r)
	b.Raw("single:").
		Raw("TESTQ $16, CX"). // ido & 1
		Raw("JZ knext")
	x.body(r, false)
	x.advance(16)
	b.Raw("knext:")
	// P advanced by S over the block; the next block starts r·S after this one.
	skNextBlock(b, r)
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
	y, x := skEmit{b: b, w: "Y", last: r}, skEmit{b: b, w: "X", last: r}
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
	z := skEmit{b: b, w: "Z", wide: true, g: 4}
	y := skEmit{b: b, w: "Y", wide: true, g: 2}
	x := skEmit{b: b, w: "X", wide: true, g: 1}
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
		Raw("MOVQ R15, R10")
	z.body(r, true)
	z.advance(64)
	z.twNext(r)
	b.Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9"). // quads = ido/4
		Raw("DECQ R9").
		Raw("iloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	z.body(r, false)
	z.advance(64)
	z.twNext(r)
	b.Raw("DECQ R9").
		Raw("JMP iloop").
		Raw("pair:").
		Raw("TESTQ $32, CX"). // ido & 2
		Raw("JZ single")
	y.body(r, false)
	y.advance(32)
	y.twNext(r)
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
	z, y, x := skEmit{b: b, w: "Z", last: r, wide: true}, skEmit{b: b, w: "Y", last: r, wide: true}, skEmit{b: b, w: "X", last: r, wide: true}
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

// genStockhamLastRun emits skLastRun{r}{AVX2,AVX512}(cc, ch, k, os, runs,
// run, gap): the final pass (ido == 1) of the fft package's blocked schedule
// (stockham.go, cascadeBlocked), which writes straight into the transform's
// output instead of a local buffer. It runs runs·run blocks, the bodies the
// final-pass kernels run (genStockhamLastAVX2/512, so the arithmetic is the
// same, operation for operation), reading the blocks contiguously; its
// outputs land in runs of run consecutive points, gap points apart, and
// output j sits os points after output j-1. run is a multiple of 4. The
// plain final-pass kernel is the case runs = 1, run = gap = os.
//
// Registers: AX = input, BX = output run cursor, DI = BX + 4·OS, DX = OS =
// os·16, R13 = 3·OS, R14 = constants, R8 = runs left, R9 = steps left in the
// run, R10 = steps per run, R11 = (gap - run)·16, the jump to the next run.
func genStockhamLastRun(f *emit.File, r int, wide bool) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "os", "runs", "run", "gap"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64, amd64.Int64, amd64.Int64}, nil, nil,
	)
	name, step, w := fmt.Sprintf("skLastRun%dAVX2", r), 2, "Y"
	if wide {
		name, step, w = fmt.Sprintf("skLastRun%dAVX512", r), 4, "Z"
	}
	b := amd64.NewFunc(name, sig, 0)
	e := skEmit{b: b, w: w, last: r, wide: wide}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("os", "DX").
		LoadArg("runs", "R8").LoadArg("run", "R10").LoadArg("gap", "R11")
	b.Raw("SHLQ $4, DX"). // OS
				Raw("LEAQ (DX)(DX*2), R13").
				Raw("SUBQ R10, R11").
				Raw("SHLQ $4, R11"). // (gap - run)·16
				Raw("SHRQ $%d, R10", map[int]int{2: 1, 4: 2}[step]).
				Raw("rloop:").
				Raw("TESTQ R8, R8").
				Raw("JZ done").
				Raw("LEAQ (BX)(DX*4), DI").
				Raw("MOVQ R10, R9").
				Raw("sloop:").
				Raw("TESTQ R9, R9").
				Raw("JZ rnext")
	e.body(r, false)
	b.Raw("ADDQ $%d, AX", 16*step*r).
		Raw("ADDQ $%d, BX", 16*step).
		Raw("ADDQ $%d, DI", 16*step).
		Raw("DECQ R9").
		Raw("JMP sloop").
		Raw("rnext:").
		Raw("ADDQ R11, BX").
		Raw("DECQ R8").
		Raw("JMP rloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genStockhamStrided emits skStrided{r}{AVX2,AVX512}(cc, ch, tw, k, cnt, nb,
// sin, sout, din, dout, dtw, bf, br): genStockhamAVX2's (or AVX512's) pass
// over nb blocks of cnt points (cnt a positive multiple of 4), with every
// stride a parameter instead of a function of ido and l1. Point i of block b
// reads its r inputs sin points apart from cc + b·bin + i and writes its r
// outputs sout points apart from ch + b·bout + i, where the caller passes the
// block strides as din = (bin - cnt)·16 and dout = (bout - cnt)·16, the jump
// left after the block's points. Block b's twiddles start dtw bytes after
// block b-1's (StockhamTwiddles' layout, so a slice of a pass's table that
// starts at a point i0 with i0 mod 4 = 0 serves points i0, i0+1, ...). bf and
// br say whether the first point of block 0, and of the other blocks, is the
// pass's untwiddled point i = 0, which the kernel then keeps unmultiplied as
// the plain pass does (the blend); any other point is multiplied. The
// butterflies and the products are the plain pass's, so a pass split into
// strided pieces is bit-identical to the pass. The fft package runs two
// passes chunk by chunk with it (cascadePair), one reading the array and the
// other writing it, through a buffer that stays in cache.
//
// Registers as genStockhamAVX2, plus R11 = whether the block's first point is
// i = 0; R8 = blocks left, R9 = four-point groups left.
func genStockhamStrided(f *emit.File, r int, wide bool) {
	names := []string{"cc", "ch", "tw", "k", "cnt", "nb", "sin", "sout", "din", "dout", "dtw", "bf", "br"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr}
	for range names[4:] {
		types = append(types, amd64.Int64)
	}
	sig := amd64.Layout(names, types, nil, nil)
	arg := func(name string) string {
		for _, p := range sig.Args {
			if p.Name == name {
				return fmt.Sprintf("%s+%d(FP)", name, p.Offset)
			}
		}
		panic(name)
	}
	name := fmt.Sprintf("skStrided%dAVX2", r)
	if wide {
		name = fmt.Sprintf("skStrided%dAVX512", r)
	}
	b := amd64.NewFunc(name, sig, 0)
	// One four-point group: a ZMM register, or two YMM pairs.
	group := func(first bool) {
		if wide {
			z := skEmit{b: b, w: "Z", wide: true, g: 4}
			z.body(r, first)
			z.advance(64)
			z.twNext(r)
			return
		}
		q0 := skEmit{b: b, w: "Y", g: 4}
		q1 := skEmit{b: b, w: "Y", g: 4, half: 1}
		q0.body(r, first)
		q0.advance(32)
		q1.body(r, false)
		q1.advance(32)
		q0.twNext(r)
	}
	if wide {
		b.Raw("MOVQ $3, R9").Raw("KMOVB R9, K1") // lanes 0–1: the i = 0 point
	}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("sin", "CX").LoadArg("sout", "DX").LoadArg("nb", "R8").LoadArg("bf", "R11")
	b.Raw("SHLQ $4, CX").
		Raw("SHLQ $4, DX").
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("kloop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("MOVQ R15, R10").
		Raw("MOVQ %s, R9", arg("cnt")).
		Raw("SHRQ $2, R9").
		Raw("TESTQ R11, R11").
		Raw("JZ plain")
	group(true)
	b.Raw("JMP rest").
		Raw("plain:")
	group(false)
	b.Raw("rest:").
		Raw("DECQ R9").
		Raw("gloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ knext")
	group(false)
	b.Raw("DECQ R9").
		Raw("JMP gloop").
		Raw("knext:").
		Raw("ADDQ %s, AX", arg("din")).
		Raw("ADDQ %s, BX", arg("dout")).
		Raw("ADDQ %s, R15", arg("dtw")).
		Raw("MOVQ %s, R11", arg("br")).
		Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genStockhamBatchAVX2 emits skBatch{r}AVX2: one Stockham pass of radix r over
// a batch of w neighbouring transforms, the lines of a non-contiguous axis of
// an N-D array. Point p of the batch's transforms is w consecutive complex
// values ("a row" of the strip), sIn bytes after point p-1 on input and sOut
// on output, so the pass reads its strip straight out of the array (first
// pass) or writes it straight back (last pass) where the per-line path
// gathers and scatters. The butterflies are the ones every other pass kernel
// runs; point i of a block multiplies output j by one twiddle for the whole
// row, so the twiddle is broadcast from memory and the product costs one
// shuffle. Point i = 0 is not multiplied at all, as in the scalar pass, so no
// blend is needed. tw holds, for i = 1 .. ido-1, the r-1 twiddles of point i
// one after the other (StockhamBatchTwiddles).
//
// Arguments, in bytes where they are strides: jin = ido·sIn (input j stride),
// jout = l1·ido·sOut (output j stride), adjin = sIn - 16·w, adjout = sOut -
// 16·w; pairs = w/2 and odd = w&1 split the row into YMM pairs and one XMM
// point. Registers: AX/SI input (SI = AX + 4·jin), BX/DI output, CX = jin, DX
// = jout, R12 = 3·CX, R13 = 3·DX, R14 = constants, R10 = twiddle cursor, R15 =
// twiddle base, R8 = blocks left, R9 = points left, R11 = pairs left.
func genStockhamBatchAVX2(f *emit.File, r int) {
	names := []string{"cc", "ch", "tw", "k", "ido", "l1", "pairs", "odd", "jin", "jout", "adjin", "adjout"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr}
	for range names[4:] {
		types = append(types, amd64.Int64)
	}
	sig := amd64.Layout(names, types, nil, nil)
	arg := func(name string) string {
		for _, p := range sig.Args {
			if p.Name == name {
				return fmt.Sprintf("%s+%d(FP)", name, p.Offset)
			}
		}
		panic(name)
	}
	b := amd64.NewFunc(fmt.Sprintf("skBatch%dAVX2", r), sig, 0)
	y0, x0 := skEmit{b: b, w: "Y", notw: true}, skEmit{b: b, w: "X", notw: true}
	yt, xt := skEmit{b: b, w: "Y", bcast: true}, skEmit{b: b, w: "X", bcast: true}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("jin", "CX").LoadArg("jout", "DX").LoadArg("l1", "R8")
	b.Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("kloop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("MOVQ R15, R10")
	// row emits one point's row: the YMM pairs, then the odd XMM point, then
	// the step to the next point's row.
	row := func(label string, y, x skEmit) {
		b.Raw("MOVQ %s, R11", arg("pairs")).
			Raw("%sp:", label).
			Raw("TESTQ R11, R11").
			Raw("JZ %so", label)
		y.body(r, false)
		y.advance(32)
		b.Raw("DECQ R11").
			Raw("JMP %sp", label).
			Raw("%so:", label).
			Raw("CMPQ %s, $0", arg("odd")).
			Raw("JEQ %se", label)
		x.body(r, false)
		x.advance(16)
		b.Raw("%se:", label).
			Raw("ADDQ %s, AX", arg("adjin")).
			Raw("ADDQ %s, SI", arg("adjin")).
			Raw("ADDQ %s, BX", arg("adjout")).
			Raw("ADDQ %s, DI", arg("adjout"))
	}
	row("zero", y0, x0) // point i = 0: no twiddle
	b.Raw("MOVQ %s, R9", arg("ido")).
		Raw("DECQ R9").
		Raw("iloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ knext")
	row("tw", yt, xt)
	b.Raw("ADDQ $%d, R10", (r-1)*16).
		Raw("DECQ R9").
		Raw("JMP iloop").
		Raw("knext:")
	// P advanced by jin over the block; the next block starts r·jin after it.
	skNextBlock(b, r)
	// O advanced by ido·sOut, which is exactly the next block's output start.
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// ---------------------------------------------------------------------------
// Radix-16 Stockham passes (the fft package's radix16.go, Round 19).
//
// The 16-point butterfly is the 4×4 split of bfly16, the algorithm of FFTW's
// n1fv_16/t1fv_16 codelets: four radix-4 butterflies over the inputs n2,
// n2+4, n2+8, n2+12 (stage 1), the internal twiddles W16^(n2·k1), then four
// radix-4 butterflies over the stage-1 outputs k1 of each (stage 2), whose
// output k2 is the pass's output k1+4·k2. Every operation is the Go
// reference's, in the same order, separately rounded.
//
// Sixteen YMM registers cannot hold the sixteen stage-1 values and the
// working set, so the values of k1 = 2 and 3 (eight registers) are spilled
// to the frame, 32 bytes each, and read back for stage 2; k1 = 0 and 1 stay
// in Y8..Y15.
//
// The twiddles come pre-duplicated (radix16Twiddles): each product takes
// (wr, wr) and (wi, wi) as memory operands, and swapping y is its only
// shuffle. On Zen 3 that made the twiddled pass 7% faster (2026-10-05).
//
// The pass kernel addresses sixteen input and sixteen output streams through
// four base registers each (skIn, skOut): AX, SI, R8, R9 in and BX, DI, R11,
// R15 out. That leaves no register for the block and group counters, which
// live in the frame (radix16Blocks, radix16Groups), nor for the twiddle base,
// which is reloaded from the arguments at each block.
//
// There is no AVX-512 radix-16 kernel: none could be verified on hardware
// this round, and the fft package does not factor with radix 16 where the
// AVX-512 kernels run.

const (
	radix16Spill  = 256 // eight spilled stage-1 values, 32 bytes each
	radix16Blocks = 256 // blocks left (pass kernel)
	radix16Groups = 264 // four-point groups left (pass kernel)
	radix16Frame  = 272
)

// r16keep is the register that holds stage-1 value (n2, k1), k1 = 0 or 1,
// between the stages.
func r16keep(n2, k1 int) int { return 8 + 4*k1 + n2 }

// r16slot is the frame offset of a spilled stage-1 value (k1 = 2, 3).
func r16slot(n2, k1 int) string { return fmt.Sprintf("%d(SP)", 32*(4*(k1-2)+n2)) }

// rot16 emits dst = src·W16^e for e in {1, 2, 3, 6, 9}: c·src + swap(src)·[−ss,
// ss], the four operations of radix16Rot. ta and tb are scratch.
func (e skEmit) rot16(dst, src, ex, ta, tb int) {
	k := map[int][2]int{
		1: {kR16C1, kR16SS1},
		2: {kH, kR16SSH},
		3: {kR16S1, kR16SSC1},
		6: {kR16NH, kR16SSH},
		9: {kR16NC1, kR16SSN1},
	}[ex]
	e.mulk(ta, src, k[0])
	e.perm("swap", src, tb)
	e.mulk(tb, tb, k[1])
	e.add(dst, ta, tb)
}

// bfly4r emits the radix-4 butterfly of registers in into z, through t: the
// Go bfly4's t2, t1, t3, t4 = rotS(x1 − x3), then y0..y3.
func (e skEmit) bfly4r(in, t, z [4]int) {
	e.add(t[0], in[0], in[2])
	e.sub(t[1], in[0], in[2])
	e.add(t[2], in[1], in[3])
	e.sub(t[3], in[1], in[3])
	e.rot(t[3], t[3])
	e.add(z[0], t[0], t[2])
	e.add(z[1], t[1], t[3])
	e.sub(z[2], t[0], t[2])
	e.sub(z[3], t[1], t[3])
}

// body16 emits one radix-16 butterfly over the points the register width
// holds, with the outer twiddles (twStore) unless this is a final pass.
func (e skEmit) body16(first bool) {
	// Stage 1.
	for n2 := 0; n2 < 4; n2++ {
		for q := 0; q < 4; q++ {
			e.in(q, n2+4*q)
		}
		// y0 straight into its register; y1, y2, y3 into Y0..Y2.
		e.bfly4r([4]int{0, 1, 2, 3}, [4]int{4, 5, 6, 7}, [4]int{r16keep(n2, 0), 0, 1, 2})
		for k1 := 1; k1 < 4; k1++ {
			src, dst := k1-1, 5 // y_k1, and where its twiddled value goes
			if k1 == 1 {
				dst = r16keep(n2, 1)
			}
			switch ex := n2 * k1; ex {
			case 0:
				if k1 == 1 {
					e.raw("VMOVAPD %s, %s", e.v(src), e.v(dst))
				}
				dst = src
			case 4:
				e.rot(dst, src)
			default:
				e.rot16(dst, src, ex, 3, 4)
			}
			if k1 >= 2 {
				e.raw("VMOVUPD %s, %s", e.v(dst), r16slot(n2, k1))
			}
		}
	}
	// Stage 2.
	for k1 := 0; k1 < 4; k1++ {
		var in [4]int
		t, z, tt := [4]int{0, 1, 2, 3}, [4]int{4, 5, 6, 7}, [4]int{8, 9, 10, 11}
		if k1 >= 2 {
			for n2 := 0; n2 < 4; n2++ {
				in[n2] = n2
				e.raw("VMOVUPD %s, %s", r16slot(n2, k1), e.v(n2))
			}
			t, z, tt = [4]int{4, 5, 6, 7}, [4]int{8, 9, 10, 11}, [4]int{12, 13, 14, 15}
		} else {
			for n2 := 0; n2 < 4; n2++ {
				in[n2] = r16keep(n2, k1)
			}
		}
		e.bfly4r(in, t, z)
		for k2 := 0; k2 < 4; k2++ {
			if j := k1 + 4*k2; j == 0 {
				e.st(z[0], skOut(0))
			} else {
				e.twStore(z[k2], j, tt[0], tt[1], tt[2], tt[3], first)
			}
		}
	}
}

// advance16 moves the eight stream bases of the radix-16 pass kernel.
func (e skEmit) advance16(bytes int) {
	for _, r := range []string{"AX", "SI", "R8", "R9", "BX", "DI", "R11", "R15"} {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

// genStockham16 emits skPass16AVX2: the radix-16 pass, the loop of
// genStockhamAVX2 around body16.
func genStockham16(f *emit.File) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("skPass16AVX2", sig, radix16Frame)
	q0 := skEmit{b: b, w: "Y", g: 4, dup: true}
	q1 := skEmit{b: b, w: "Y", g: 4, half: 1, dup: true}
	p := skEmit{b: b, w: "Y", g: 2, dup: true}
	x := skEmit{b: b, w: "X", g: 1, dup: true}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").
		LoadArg("ido", "CX").LoadArg("l1", "R8")
	b.Raw("MOVQ CX, DX").
		Raw("IMULQ R8, DX").
		Raw("SHLQ $4, CX"). // S
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("MOVQ R8, %d(SP)", radix16Blocks).
		Raw("kloop:").
		Raw("CMPQ %d(SP), $0", radix16Blocks).
		Raw("JEQ done").
		Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9"). // four-point groups = ido/4
		Raw("MOVQ R9, %d(SP)", radix16Groups).
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("LEAQ (AX)(CX*8), R8").
		Raw("LEAQ (SI)(CX*8), R9").
		Raw("LEAQ (BX)(DX*8), R11").
		Raw("LEAQ (DI)(DX*8), R15")
	b.LoadArg("tw", "R10")
	b.Raw("CMPQ %d(SP), $0", radix16Groups).
		Raw("JEQ firstpair")
	q0.body16(true)
	q0.advance16(32)
	q1.body16(false)
	q1.advance16(32)
	q0.twNext(16)
	b.Raw("DECQ %d(SP)", radix16Groups).
		Raw("qloop:").
		Raw("CMPQ %d(SP), $0", radix16Groups).
		Raw("JEQ pair")
	q0.body16(false)
	q0.advance16(32)
	q1.body16(false)
	q1.advance16(32)
	q0.twNext(16)
	b.Raw("DECQ %d(SP)", radix16Groups).
		Raw("JMP qloop").
		Raw("pair:").
		Raw("TESTQ $32, CX"). // ido & 2
		Raw("JZ single")
	p.body16(false)
	p.advance16(32)
	p.twNext(16)
	b.Raw("JMP single").
		Raw("firstpair:") // ido is 2 or 3: the first pair is the two-point group
	p.body16(true)
	p.advance16(32)
	p.twNext(16)
	b.Raw("single:").
		Raw("TESTQ $16, CX"). // ido & 1
		Raw("JZ knext")
	x.body16(false)
	x.advance16(16)
	// P advanced by S over the block; the next block starts 16·S after this
	// one. O advanced by S, which is the next block's output start.
	b.Raw("knext:").
		Raw("LEAQ (AX)(CX*8), AX").
		Raw("LEAQ (AX)(CX*4), AX").
		Raw("ADDQ R12, AX").
		Raw("DECQ %d(SP)", radix16Blocks).
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genStockham16Last emits skLast16AVX2: the final radix-16 pass (ido == 1),
// two blocks per register as in genStockhamLastAVX2, an odd l1 finished by
// one 128-bit step.
func genStockham16Last(f *emit.File) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("skLast16AVX2", sig, radix16Spill)
	y := skEmit{b: b, w: "Y", last: 16}
	x := skEmit{b: b, w: "X", last: 16}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("l1", "DX")
	b.Raw("MOVQ DX, R9").
		Raw("SHRQ $1, R9"). // block pairs
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("LEAQ (BX)(DX*8), R11").
		Raw("LEAQ (DI)(DX*8), R15").
		Raw("loop:").
		Raw("TESTQ R9, R9").
		Raw("JZ tail")
	y.body16(false)
	b.Raw("ADDQ $%d, AX", 32*16)
	for _, r := range []string{"BX", "DI", "R11", "R15"} {
		b.Raw("ADDQ $32, %s", r)
	}
	b.Raw("DECQ R9").
		Raw("JMP loop").
		Raw("tail:").
		Raw("TESTQ $16, DX"). // l1 odd
		Raw("JZ done")
	x.body16(false)
	b.Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// ---------------------------------------------------------------------------
// float32 (complex64) Stockham pass kernels, into stockham32_amd64.s.
//
// sk32Pass{r}AVX2(cc, ch, tw *complex64, k *float32, ido, l1 int) and
// sk32Last{r}AVX2(cc, ch *complex64, k *float32, l1 int) are the float32
// passes of the fft package's stockham32_passes.go (f32Pass{r},
// f32Pass{r}last) for radix 2, 3, 4, 5 and 8. They are genStockhamAVX2 and
// genStockhamLastAVX2 with single-precision instructions: a complex64 is 8
// bytes, so a YMM register holds four points and an XMM register two. They are
// kept apart from the float64 emitter (f32simdEmit, not skEmit) so a change to
// either family cannot move the other's arithmetic.
//
// Bit identity with the Go passes, which gc compiles for GOAMD64=v1 with
// separately rounded MULSS/ADDSS/SUBSS (no FMA): every operation below is the
// Go pass's, in its order, as VMULPS/VADDPS/VSUBPS; the ±i rotation is a lane
// swap (VPERMILPS) and a sign flip (VXORPS), both exact. The twiddle product
// y·w loads w's real parts duplicated (VMOVSLDUP) and its imaginary parts
// duplicated (VMOVSHDUP), both plain loads, and computes (yr·wr − yi·wi,
// yi·wr + yr·wi) with VADDSUBPS: f32Mul's products, its imaginary sum taken
// in the other order, which IEEE addition makes exact. The point i = 0, which
// the Go pass does not multiply, is multiplied by w = 1 with its group and
// then blended back (VBLENDPS $3): a multiply by one is not exact on signed
// zeros.
//
// Points are walked in twiddle groups (kernels.StockhamTwiddles32 on amd64):
// groups of four points in a YMM register, then one of two in an XMM register
// for ido&2 and one of one for ido&1, loaded and stored with VMOVSD (8 bytes)
// and computed in an XMM register whose upper lanes are zero. A group of g
// points stores its r-1 twiddle runs back to back, g entries each, so a pass
// reads its twiddles as one sequential stream (Round 10).
//
// k points at the float32 constants table (stockham32_amd64.go): 32-byte rows
// at the offsets of the float64 table (kSign … kH), each constant rounded once
// to float32 as the Go passes' untyped constants are.
//
// Registers as genStockhamAVX2's, with S = ido·8 and OS = l1·ido·8.

type f32simdEmit struct {
	b    *amd64.Builder
	w    string // "Y" (four points), "X" (two) or "S" (one, in an X register)
	last int    // radix of a final-pass kernel (ido == 1), 0 for the others
	g    int    // twiddle group size in points: 4, 2 or 1
	// notw stores every output untwiddled (point i = 0 of a batched pass,
	// Round 25); bcast multiplies output j by one twiddle broadcast to every
	// lane, the complex64 at (j-1)·8(R10) (the other points of a batched
	// pass).
	notw, bcast bool
}

func (e f32simdEmit) v(i int) string {
	if e.w == "Y" {
		return fmt.Sprintf("Y%d", i)
	}
	return fmt.Sprintf("X%d", i)
}

func (e f32simdEmit) raw(format string, a ...any) { e.b.Raw(format, a...) }

func f32simdIn(j int) string {
	return [...]string{"(AX)", "(AX)(CX*1)", "(AX)(CX*2)", "(AX)(R12*1)", "(SI)", "(SI)(CX*1)", "(SI)(CX*2)", "(SI)(R12*1)"}[j]
}

func f32simdOut(j int) string {
	return [...]string{"(BX)", "(BX)(DX*1)", "(BX)(DX*2)", "(BX)(R13*1)", "(DI)", "(DI)(DX*1)", "(DI)(DX*2)", "(DI)(R13*1)"}[j]
}

func (e f32simdEmit) ld(dst int, addr string) {
	if e.w == "S" {
		e.raw("VMOVSD %s, X%d", addr, dst)
		return
	}
	e.raw("VMOVUPS %s, %s", addr, e.v(dst))
}

func (e f32simdEmit) st(src int, addr string) {
	if e.w == "S" {
		e.raw("VMOVSD X%d, %s", src, addr)
		return
	}
	e.raw("VMOVUPS %s, %s", e.v(src), addr)
}

// in loads input j. In a final pass (ido == 1) a register holds blocks k ..
// k+3 (or k, k+1, or k alone) instead of points: input j of block k+q is
// 8·(q·r+j) bytes into the blocks, gathered two by two (VMOVSD, VMOVHPS) and
// joined with VINSERTF128 through X15, which no final-pass body uses.
func (e f32simdEmit) in(dst, j int) {
	if e.last == 0 {
		e.ld(dst, f32simdIn(j))
		return
	}
	r := e.last
	e.raw("VMOVSD %d(AX), X%d", 8*j, dst)
	if e.w == "S" {
		return
	}
	e.raw("VMOVHPS %d(AX), X%d, X%d", 8*(r+j), dst, dst)
	if e.w == "X" {
		return
	}
	e.raw("VMOVSD %d(AX), X15", 8*(2*r+j))
	e.raw("VMOVHPS %d(AX), X15, X15", 8*(3*r+j))
	e.raw("VINSERTF128 $1, X15, Y%d, Y%d", dst, dst)
}

func (e f32simdEmit) add(dst, a, b int) { e.raw("VADDPS %s, %s, %s", e.v(b), e.v(a), e.v(dst)) }
func (e f32simdEmit) sub(dst, a, b int) { e.raw("VSUBPS %s, %s, %s", e.v(b), e.v(a), e.v(dst)) }
func (e f32simdEmit) mulk(dst, a int, off int) {
	e.raw("VMULPS %d(R14), %s, %s", off, e.v(a), e.v(dst))
}

// swap exchanges the real and imaginary part of every point.
func (e f32simdEmit) swap(src, dst int) { e.raw("VPERMILPS $0xB1, %s, %s", e.v(src), e.v(dst)) }

// rot: dst = rotS(src), a swap and the sign flip of the direction's row.
func (e f32simdEmit) rot(dst, src int) {
	e.swap(src, dst)
	e.raw("VXORPS %d(R14), %s, %s", kSign, e.v(dst), e.v(dst))
}

func (e f32simdEmit) tw(j int) string { return fmt.Sprintf("%d(R10)", (j-1)*8*e.g) }

func (e f32simdEmit) twNext(r int) { e.raw("ADDQ $%d, R10", (r-1)*8*e.g) }

func (e f32simdEmit) advance(bytes int) {
	for _, r := range []string{"AX", "SI", "BX", "DI"} {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

// twStore multiplies y by twiddle j and stores it to output j; see the
// section comment. A lone point loads its twiddle with VMOVSD (a 16-byte
// VMOVSLDUP would read past the table) and duplicates it in registers.
func (e f32simdEmit) twStore(y, j, t1, t2, t3, dst int, first bool) {
	if e.last != 0 || e.notw {
		e.st(y, f32simdOut(j))
		return
	}
	if e.bcast {
		e.raw("VBROADCASTSS %d(R10), %s", (j-1)*8, e.v(t1))
		e.raw("VBROADCASTSS %d(R10), %s", (j-1)*8+4, e.v(t2))
	} else if e.w == "S" {
		e.raw("VMOVSD %s, X%d", e.tw(j), t1)
		e.raw("VMOVSHDUP X%d, X%d", t1, t2)
		e.raw("VMOVSLDUP X%d, X%d", t1, t1)
	} else {
		e.raw("VMOVSLDUP %s, %s", e.tw(j), e.v(t1))
		e.raw("VMOVSHDUP %s, %s", e.tw(j), e.v(t2))
	}
	e.swap(y, t3)
	e.raw("VMULPS %s, %s, %s", e.v(y), e.v(t1), e.v(t1))
	e.raw("VMULPS %s, %s, %s", e.v(t3), e.v(t2), e.v(t2))
	e.raw("VADDSUBPS %s, %s, %s", e.v(t2), e.v(t1), e.v(dst))
	if first {
		e.raw("VBLENDPS $3, %s, %s, %s", e.v(y), e.v(dst), e.v(dst))
	}
	e.st(dst, f32simdOut(j))
}

// body is skEmit.body's butterflies, operation for operation, which are also
// stockham32_passes.go's.
func (e f32simdEmit) body(r int, first bool) {
	switch r {
	case 2:
		e.in(0, 0)
		e.in(1, 1)
		e.add(2, 0, 1)
		e.st(2, f32simdOut(0))
		e.sub(3, 0, 1)
		e.twStore(3, 1, 5, 6, 7, 4, first)
	case 3:
		e.in(0, 0)
		e.in(1, 1)
		e.in(2, 2)
		e.add(3, 1, 2) // t1
		e.sub(4, 1, 2) // t2
		e.add(5, 0, 3)
		e.st(5, f32simdOut(0))
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
		e.st(0, f32simdOut(0))
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
		e.st(7, f32simdOut(0))
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
		e.st(4, f32simdOut(0))
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

// f32simdNextBlock advances P from the end of one block (it moved by S) to
// the start of the next, r·S after this one's.
func f32simdNextBlock(b *amd64.Builder, r int) {
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

// genF32StockhamAVX2 emits sk32Pass{r}AVX2 (ido >= 2).
func genF32StockhamAVX2(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("sk32Pass%dAVX2", r), sig, 0)
	y := f32simdEmit{b: b, w: "Y", g: 4}
	x := f32simdEmit{b: b, w: "X", g: 2}
	s := f32simdEmit{b: b, w: "S", g: 1}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("ido", "CX").LoadArg("l1", "R8")
	b.Raw("MOVQ CX, DX").
		Raw("IMULQ R8, DX").
		Raw("SHLQ $3, CX"). // S
		Raw("SHLQ $3, DX"). // OS
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("kloop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("MOVQ R15, R10").
		Raw("MOVQ CX, R9").
		Raw("SHRQ $5, R9"). // four-point groups = ido/4
		Raw("TESTQ R9, R9").
		Raw("JZ firstpair")
	y.body(r, true)
	y.advance(32)
	y.twNext(r)
	b.Raw("DECQ R9").
		Raw("qloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	y.body(r, false)
	y.advance(32)
	y.twNext(r)
	b.Raw("DECQ R9").
		Raw("JMP qloop").
		Raw("pair:").
		Raw("TESTQ $16, CX"). // ido & 2
		Raw("JZ single")
	x.body(r, false)
	x.advance(16)
	x.twNext(r)
	b.Raw("JMP single").
		Raw("firstpair:") // ido is 2 or 3: the first group is the pair
	x.body(r, true)
	x.advance(16)
	x.twNext(r)
	b.Raw("single:").
		Raw("TESTQ $8, CX"). // ido & 1
		Raw("JZ knext")
	s.body(r, false)
	s.advance(8)
	b.Raw("knext:")
	f32simdNextBlock(b, r)
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genF32StockhamLastAVX2 emits sk32Last{r}AVX2: the final pass (ido == 1),
// four blocks per YMM register, then two and one for l1 mod 4. Output j of
// blocks k .. k+3 is contiguous, one VMOVUPS.
func genF32StockhamLastAVX2(f *emit.File, r int) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("sk32Last%dAVX2", r), sig, 0)
	y := f32simdEmit{b: b, w: "Y", last: r}
	x := f32simdEmit{b: b, w: "X", last: r}
	s := f32simdEmit{b: b, w: "S", last: r}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("l1", "DX")
	b.Raw("MOVQ DX, R9").
		Raw("SHRQ $2, R9"). // groups of four blocks
		Raw("SHLQ $3, DX"). // OS
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("loop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	y.body(r, false)
	b.Raw("ADDQ $%d, AX", 32*r).
		Raw("ADDQ $32, BX").
		Raw("ADDQ $32, DI").
		Raw("DECQ R9").
		Raw("JMP loop").
		Raw("pair:").
		Raw("TESTQ $16, DX"). // l1 & 2
		Raw("JZ single")
	x.body(r, false)
	b.Raw("ADDQ $%d, AX", 16*r).
		Raw("ADDQ $16, BX").
		Raw("ADDQ $16, DI").
		Raw("single:").
		Raw("TESTQ $8, DX"). // l1 & 1
		Raw("JZ done")
	s.body(r, false)
	b.Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genF32StockhamFile writes the float32 pass kernels.
func genF32StockhamFile() {
	f := emit.NewFile("amd64")
	for _, r := range []int{2, 3, 4, 5, 8} {
		genF32StockhamAVX2(f, r)
	}
	for _, r := range []int{2, 3, 4, 5, 8} {
		genF32StockhamLastAVX2(f, r)
	}
	writeFile("stockham32_amd64.s", f.String())
}

// ---------------------------------------------------------------------------
// Split-layout Stockham passes (Round 23).
//
// A power of two keeps its data block-split between passes: four points
// p = 4q .. 4q+3 occupy the 64 bytes they occupy interleaved, as the real
// parts re(4q), re(4q+2), re(4q+1), re(4q+3), then the imaginary parts in the
// same order. That order is what VUNPCKLPD and VUNPCKHPD make of two
// interleaved YMM loads ([r0 i0 r1 i1], [r2 i2 r3 i3] -> [r0 r2 r1 r3] and
// [i0 i2 i1 i3]), one in-lane shuffle per register, and they undo it the
// same way. Inside a pass the points of a block are independent, so the
// order within the block does not matter as long as the twiddles are stored
// in it too (kernels.StockhamSplitTwiddles).
//
// With the parts apart, a YMM register holds one part of four points: the
// ±i rotations are a choice of operands (no shuffle, no sign flip), and a
// twiddle product is four multiplies and two adds per four points with no
// shuffle, where the interleaved kernels spend two shuffles per two points.
//
// skSplit{r}{in}{out}{Fwd|Inv}(cc, ch, tw *complex128, k *float64, ido, l1
// int) runs one whole pass (ido a multiple of four) reading cc interleaved
// (I) or split (S) and writing ch the same way. The direction is compiled in:
// the rotations differ in which operand is added and which subtracted. The
// arithmetic is stockham.go's, operation for operation, separately rounded,
// so the result is the scalar pass's bits: every output is the same IEEE
// operation on the same operands, a − (−b) and a + (−b) being a + b and
// a − b exactly. Point i = 0 of a block (lane 0 of its first group) is
// multiplied by 1 with its group and then put back by a blend, as in the
// interleaved kernels.
//
// Registers as in genStockhamAVX2: AX/SI inputs, BX/DI outputs, CX = S, DX =
// OS, R12 = 3·S, R13 = 3·OS, R8 blocks left, R9 groups left, R10 twiddle
// cursor, R15 twiddle base, R14 constants (kernels.splitConst).

// Offsets into kernels.splitConst.
const (
	splitKH   = 0  // √2/2
	splitKNeg = 32 // −0: the sign flip
)

type splitEmit struct {
	b         *amd64.Builder
	r         int
	inS, outS bool // input / output split
	inverse   bool
}

func (e splitEmit) raw(format string, a ...any) { e.b.Raw(format, a...) }

// y names a YMM register.
func y(i int) string { return fmt.Sprintf("Y%d", i) }

// at32 is addr 32 bytes further on: "(AX)(CX*1)" -> "32(AX)(CX*1)".
func at32(addr string) string { return "32" + addr }

// load puts input stream j's four points, split, into re and im; t0 and t1
// are scratch for an interleaved input.
func (e splitEmit) load(j, re, im, t0, t1 int) {
	a := skIn(j)
	if e.inS {
		e.raw("VMOVUPD %s, %s", a, y(re))
		e.raw("VMOVUPD %s, %s", at32(a), y(im))
		return
	}
	e.raw("VMOVUPD %s, %s", a, y(t0))
	e.raw("VMOVUPD %s, %s", at32(a), y(t1))
	e.raw("VUNPCKLPD %s, %s, %s", y(t1), y(t0), y(re))
	e.raw("VUNPCKHPD %s, %s, %s", y(t1), y(t0), y(im))
}

// store writes re, im to output stream j; t0 and t1 are scratch for an
// interleaved output.
func (e splitEmit) store(j, re, im, t0, t1 int) {
	a := skOut(j)
	if e.outS {
		e.raw("VMOVUPD %s, %s", y(re), a)
		e.raw("VMOVUPD %s, %s", y(im), at32(a))
		return
	}
	e.raw("VUNPCKLPD %s, %s, %s", y(im), y(re), y(t0))
	e.raw("VUNPCKHPD %s, %s, %s", y(im), y(re), y(t1))
	e.raw("VMOVUPD %s, %s", y(t0), a)
	e.raw("VMOVUPD %s, %s", y(t1), at32(a))
}

func (e splitEmit) add(dst, a, b int) { e.raw("VADDPD %s, %s, %s", y(b), y(a), y(dst)) }
func (e splitEmit) sub(dst, a, b int) { e.raw("VSUBPD %s, %s, %s", y(b), y(a), y(dst)) }

// twStore multiplies (re, im) by twiddle j and stores it to output j:
// (yr·wr − yi·wi, yi·wr + yr·wi), the scalar product with its imaginary sum
// in the other order, which IEEE addition makes exact. t holds four scratch
// registers. In the first group of a block, lane 0 (point i = 0) keeps the
// unmultiplied value.
func (e splitEmit) twStore(re, im, j int, t [4]int, first bool) {
	off := (j - 1) * 64
	e.raw("VMOVUPD %d(R10), %s", off, y(t[0]))    // wr
	e.raw("VMOVUPD %d(R10), %s", off+32, y(t[1])) // wi
	e.raw("VMULPD %s, %s, %s", y(t[0]), y(re), y(t[2]))
	e.raw("VMULPD %s, %s, %s", y(t[1]), y(im), y(t[3]))
	e.sub(t[2], t[2], t[3]) // yr·wr − yi·wi
	e.raw("VMULPD %s, %s, %s", y(t[0]), y(im), y(t[0]))
	e.raw("VMULPD %s, %s, %s", y(t[1]), y(re), y(t[1]))
	e.add(t[0], t[0], t[1]) // yi·wr + yr·wi
	if first {
		e.raw("VBLENDPD $1, %s, %s, %s", y(re), y(t[2]), y(t[2]))
		e.raw("VBLENDPD $1, %s, %s, %s", y(im), y(t[0]), y(t[0]))
	}
	e.store(j, t[2], t[0], t[1], t[3])
}

// rotAdd emits the pair (a + rotS(b), a − rotS(b)) on split registers:
// rotS(b) = (−s·bi, s·br), so forward (s = −1) a + rotS(b) = (ar + bi,
// ai − br) and a − rotS(b) = (ar − bi, ai + br); inverse the other way.
// pr, pi receive a + rotS(b), mr, mi a − rotS(b).
func (e splitEmit) rotAdd(pr, pi, mr, mi, ar, ai, br, bi int) {
	if !e.inverse {
		e.add(pr, ar, bi)
		e.sub(pi, ai, br)
		e.sub(mr, ar, bi)
		e.add(mi, ai, br)
		return
	}
	e.sub(pr, ar, bi)
	e.add(pi, ai, br)
	e.add(mr, ar, bi)
	e.sub(mi, ai, br)
}

func (e splitEmit) body(first bool) {
	switch e.r {
	case 4:
		// x0 (0,1) x1 (2,3) x2 (4,5) x3 (6,7), scratch 12, 13.
		for j := 0; j < 4; j++ {
			e.load(j, 2*j, 2*j+1, 12, 13)
		}
		e.add(8, 0, 4) // t2 = x0 + x2
		e.sub(0, 0, 4) // t1 = x0 − x2
		e.add(9, 1, 5)
		e.sub(1, 1, 5)
		e.add(4, 2, 6) // t3 = x1 + x3
		e.sub(2, 2, 6) // t4 = x1 − x3
		e.add(5, 3, 7)
		e.sub(3, 3, 7)
		e.add(6, 8, 4) // y0 = t2 + t3
		e.add(7, 9, 5)
		e.store(0, 6, 7, 10, 11)
		e.sub(8, 8, 4) // y2 = t2 − t3
		e.sub(9, 9, 5)
		e.rotAdd(4, 5, 0, 1, 0, 1, 2, 3) // y1, y3 = t1 ± rotS(t4)
		t := [4]int{10, 11, 12, 13}
		e.twStore(4, 5, 1, t, first)
		e.twStore(8, 9, 2, t, first)
		e.twStore(0, 1, 3, t, first)
	case 8:
		e.body8(first)
	}
}

// The radix-8 body spills a1 and a3 (the odd half's sums) to its frame
// while the even half runs: sixteen registers hold the odd half's four
// results and the even half's eight inputs only without them.
const splitFrame8 = 128

// h8 multiplies a register by √2/2 in place.
func (e splitEmit) h8(r int) { e.raw("VMULPD %d(R14), %s, %s", splitKH, y(r), y(r)) }

// neg flips a register's signs in place: −x, exactly.
func (e splitEmit) neg(r int) { e.raw("VXORPD %d(R14), %s, %s", splitKNeg, y(r), y(r)) }

// body8 is bfly8 (stockham.go's pass8) on split registers.
func (e splitEmit) body8(first bool) {
	// Odd half. x1 (0,1), x5 (2,3), then x3 (3,5), x7 (6,7).
	e.load(1, 0, 1, 12, 13)
	e.load(5, 2, 3, 12, 13)
	e.add(4, 0, 2) // a1 = x1 + x5
	e.sub(0, 0, 2) // a5 = x1 − x5
	e.add(2, 1, 3)
	e.sub(1, 1, 3) // a1 (4,2), a5 (0,1)
	e.load(3, 3, 5, 12, 13)
	e.load(7, 6, 7, 12, 13)
	e.add(8, 3, 6) // a3 = x3 + x7
	e.sub(3, 3, 6) // a7 = x3 − x7
	e.add(6, 5, 7)
	e.sub(5, 5, 7) // a3 (8,6), a7 (3,5)
	e.add(7, 4, 8) // a1 + a3
	e.sub(4, 4, 8) // a1 − a3
	e.add(8, 2, 6)
	e.sub(2, 2, 6) // a1 (7,8), a3 (4,2), before rotS(a3)
	e.raw("VMOVUPD %s, 0(SP)", y(7))
	e.raw("VMOVUPD %s, 32(SP)", y(8))
	e.raw("VMOVUPD %s, 64(SP)", y(4))
	e.raw("VMOVUPD %s, 96(SP)", y(2))
	e.rotAdd(6, 9, 0, 1, 0, 1, 3, 5) // a5, a7 = a5 ± rotS(a7): a5 (6,9), a7 (0,1)
	if !e.inverse {
		// a5 = h·(a5r + a5i, a5i − a5r); a7 = h·(a7i − a7r, −a7r − a7i).
		e.add(3, 6, 9)
		e.sub(5, 9, 6)
		e.sub(6, 1, 0)
		e.neg(0)
		e.sub(0, 0, 1)
	} else {
		// a5 = h·(a5r − a5i, a5i + a5r); a7 = h·(−a7i − a7r, a7r − a7i).
		e.sub(3, 6, 9)
		e.add(5, 9, 6)
		e.sub(2, 0, 1)
		e.neg(1)
		e.sub(6, 1, 0)
		e.raw("VMOVAPD %s, %s", y(2), y(0))
	}
	e.h8(3)
	e.h8(5)
	e.h8(6)
	e.h8(0) // a5 (3,5), a7 (6,0)
	// Even half. x0 (1,2), x4 (4,7), then x2 (7,9), x6 (10,11).
	e.load(0, 1, 2, 13, 14)
	e.load(4, 4, 7, 13, 14)
	e.add(8, 1, 4) // a0 = x0 + x4
	e.sub(1, 1, 4) // a4 = x0 − x4
	e.add(4, 2, 7)
	e.sub(2, 2, 7) // a0 (8,4), a4 (1,2)
	e.load(2, 7, 9, 13, 14)
	e.load(6, 10, 11, 13, 14)
	e.add(12, 7, 10) // a2 = x2 + x6
	e.sub(7, 7, 10)  // a6 = x2 − x6
	e.add(10, 9, 11)
	e.sub(9, 9, 11) // a2 (12,10), a6 (7,9)
	e.add(11, 8, 12)
	e.sub(8, 8, 12)
	e.add(12, 4, 10)
	e.sub(4, 4, 10)                    // a0 (11,12), a2 (8,4)
	e.rotAdd(10, 13, 1, 2, 1, 2, 7, 9) // a4, a6 = a4 ± rotS(a6): a4 (10,13), a6 (1,2)
	t := [4]int{3, 5, 14, 15}
	// y1, y5 = a4 ± a5.
	e.add(7, 10, 3)
	e.sub(10, 10, 3)
	e.add(9, 13, 5)
	e.sub(13, 13, 5)
	e.twStore(7, 9, 1, t, first)
	e.twStore(10, 13, 5, t, first)
	// y3, y7 = a6 ± a7.
	t = [4]int{3, 5, 10, 13}
	e.add(7, 1, 6)
	e.sub(1, 1, 6)
	e.add(9, 2, 0)
	e.sub(2, 2, 0)
	e.twStore(7, 9, 3, t, first)
	e.twStore(1, 2, 7, t, first)
	// y0, y4 = a0 ± a1.
	t = [4]int{0, 1, 2, 3}
	e.raw("VADDPD 0(SP), %s, %s", y(11), y(7))
	e.raw("VSUBPD 0(SP), %s, %s", y(11), y(11))
	e.raw("VADDPD 32(SP), %s, %s", y(12), y(9))
	e.raw("VSUBPD 32(SP), %s, %s", y(12), y(12))
	e.store(0, 7, 9, 5, 6)
	e.twStore(11, 12, 4, t, first)
	// y2, y6 = a2 ± rotS(a3).
	e.raw("VMOVUPD 64(SP), %s", y(6))
	e.raw("VMOVUPD 96(SP), %s", y(0))
	e.rotAdd(7, 9, 8, 4, 8, 4, 6, 0)
	t = [4]int{1, 2, 3, 5}
	e.twStore(7, 9, 2, t, first)
	e.twStore(8, 4, 6, t, first)
}

func (e splitEmit) advance(bytes int) {
	regs := []string{"AX", "BX"}
	if e.r > 4 {
		regs = append(regs, "SI", "DI")
	}
	for _, r := range regs {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

// splitName is the kernel's symbol.
func splitName(r int, inS, outS, inverse bool) string {
	io := map[bool]string{false: "I", true: "S"}
	dir := "Fwd"
	if inverse {
		dir = "Inv"
	}
	return fmt.Sprintf("skSplit%d%s%s%s", r, io[inS], io[outS], dir)
}

// genStockhamSplit emits one split-layout pass kernel (see above). ido is a
// multiple of four (the caller checks), so every block is whole groups of
// four points and there is no tail.
func genStockhamSplit(f *emit.File, r int, inS, outS, inverse bool) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	frame := 0
	if r == 8 {
		frame = splitFrame8
	}
	b := amd64.NewFunc(splitName(r, inS, outS, inverse), sig, frame)
	e := splitEmit{b: b, r: r, inS: inS, outS: outS, inverse: inverse}
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
		Raw("JZ done")
	if r > 4 {
		b.Raw("LEAQ (AX)(CX*4), SI").
			Raw("LEAQ (BX)(DX*4), DI")
	}
	b.Raw("MOVQ R15, R10").
		Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9") // groups = ido/4
	e.body(true)
	e.advance(64)
	b.Raw("ADDQ $%d, R10", (r-1)*64).
		Raw("DECQ R9").
		Raw("gloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ knext")
	e.body(false)
	e.advance(64)
	b.Raw("ADDQ $%d, R10", (r-1)*64).
		Raw("DECQ R9").
		Raw("JMP gloop").
		Raw("knext:")
	skNextBlock(b, r)
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genSplitFile writes the split-layout pass kernels: every radix with a
// split body, every pair of input and output layouts, both directions.
func genSplitFile() {
	f := emit.NewFile("amd64")
	for _, r := range []int{4, 8} {
		for _, inverse := range []bool{false, true} {
			for _, io := range [][2]bool{{false, true}, {true, true}, {true, false}, {false, false}} {
				genStockhamSplit(f, r, io[0], io[1], inverse)
			}
		}
	}
	writeFile("stockhamsplit_amd64.s", f.String())
}

// ---------------------------------------------------------------------------
// Radix-12 pass kernels (Round 24): the fft package's radix12.go, the
// prime-factor algorithm for 12 = 3·4. Input n = (4·n1 + 3·n2) mod 12 and
// output k = (4·k1 + 9·k2) mod 12: four radix-3 butterflies (one per n2, over
// n1), then three radix-4 butterflies (one per k1, over n2), with no twiddle
// between them. Every operation is the Go pass's (bfly3, bfly4, then the outer
// twiddle product of twStore), separately rounded, so the kernels are
// bit-identical to it as the other AVX2 kernels are to theirs.
//
// Registers: the stage-1 values of k1 = 0 stay in Y8..Y11; those of k1 = 1
// and 2 (eight registers) are spilled to the frame and read back for stage 2.
// Twelve input streams through AX, SI, R8 and twelve output streams through
// BX, DI, R11 (skIn, skOut), R10 the twiddle cursor, R15 the twiddle base, R9
// the groups left; the blocks left live in the frame.

const (
	comp12Spill  = 256 // eight spilled stage-1 values, 32 bytes each
	comp12Blocks = 256 // blocks left (pass kernel)
	comp12Frame  = 264
)

// comp12In and comp12Out are radix12.go's index maps.
var (
	comp12In  = [4][3]int{{0, 4, 8}, {3, 7, 11}, {6, 10, 2}, {9, 1, 5}}
	comp12Out = [3][4]int{{0, 9, 6, 3}, {4, 1, 10, 7}, {8, 5, 2, 11}}
)

// comp12Slot is the frame offset of the spilled stage-1 value (n2, k1), k1 =
// 1 or 2.
func comp12Slot(n2, k1 int) string { return fmt.Sprintf("%d(SP)", 32*(4*(k1-1)+n2)) }

// comp12Bfly3 emits the Go bfly3 on registers x into y, through t (three
// scratch registers): t1 = x1 + x2, t2 = x1 − x2, y0 = x0 + t1, ca = x0 −
// 0.5·t1, cb = rotS(sin(2π/3)·t2), y1 = ca + cb, y2 = ca − cb.
func (e skEmit) comp12Bfly3(x, y, t [3]int) {
	e.add(t[0], x[1], x[2]) // t1
	e.sub(t[1], x[1], x[2]) // t2
	e.add(y[0], x[0], t[0])
	e.mulk(t[2], t[0], kHalf)
	e.sub(t[2], x[0], t[2]) // ca
	e.mulk(t[1], t[1], kSin120)
	e.rot(t[1], t[1]) // cb
	e.add(y[1], t[2], t[1])
	e.sub(y[2], t[2], t[1])
}

// comp12Body emits one radix-12 butterfly over the points the register width
// holds, with the outer twiddles (twStore) unless this is a final pass.
func (e skEmit) comp12Body(first bool) {
	for n2 := 0; n2 < 4; n2++ {
		in := comp12In[n2]
		for q := 0; q < 3; q++ {
			e.in(q, in[q])
		}
		e.comp12Bfly3([3]int{0, 1, 2}, [3]int{8 + n2, 3, 4}, [3]int{5, 6, 7})
		e.raw("VMOVUPD %s, %s", e.v(3), comp12Slot(n2, 1))
		e.raw("VMOVUPD %s, %s", e.v(4), comp12Slot(n2, 2))
	}
	for k1 := 0; k1 < 3; k1++ {
		in, t, z := [4]int{8, 9, 10, 11}, [4]int{0, 1, 2, 3}, [4]int{4, 5, 6, 7}
		if k1 > 0 {
			in, t, z = [4]int{0, 1, 2, 3}, [4]int{4, 5, 6, 7}, [4]int{8, 9, 10, 11}
			for n2 := 0; n2 < 4; n2++ {
				e.raw("VMOVUPD %s, %s", comp12Slot(n2, k1), e.v(n2))
			}
		}
		e.bfly4r(in, t, z)
		for k2 := 0; k2 < 4; k2++ {
			if j := comp12Out[k1][k2]; j == 0 {
				e.st(z[k2], skOut(0))
			} else {
				e.twStore(z[k2], j, 12, 13, 14, 15, first)
			}
		}
	}
}

// comp12Advance moves the six stream bases of the radix-12 pass kernel.
func (e skEmit) comp12Advance(bytes int) {
	for _, r := range []string{"AX", "SI", "R8", "BX", "DI", "R11"} {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

// genComp12 emits skPass12AVX2: the radix-12 pass, the loop of
// genStockhamAVX2 around comp12Body.
func genComp12(f *emit.File) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("skPass12AVX2", sig, comp12Frame)
	q0 := skEmit{b: b, w: "Y", g: 4}
	q1 := skEmit{b: b, w: "Y", g: 4, half: 1}
	p := skEmit{b: b, w: "Y", g: 2}
	x := skEmit{b: b, w: "X", g: 1}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("ido", "CX").LoadArg("l1", "R8")
	b.Raw("MOVQ CX, DX").
		Raw("IMULQ R8, DX").
		Raw("SHLQ $4, CX"). // S
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("MOVQ R8, %d(SP)", comp12Blocks).
		Raw("kloop:").
		Raw("CMPQ %d(SP), $0", comp12Blocks).
		Raw("JEQ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (AX)(CX*8), R8").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("LEAQ (BX)(DX*8), R11").
		Raw("MOVQ R15, R10").
		Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9"). // four-point groups = ido/4
		Raw("TESTQ R9, R9").
		Raw("JZ firstpair")
	q0.comp12Body(true)
	q0.comp12Advance(32)
	q1.comp12Body(false)
	q1.comp12Advance(32)
	q0.twNext(12)
	b.Raw("DECQ R9").
		Raw("qloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ pair")
	q0.comp12Body(false)
	q0.comp12Advance(32)
	q1.comp12Body(false)
	q1.comp12Advance(32)
	q0.twNext(12)
	b.Raw("DECQ R9").
		Raw("JMP qloop").
		Raw("pair:").
		Raw("TESTQ $32, CX"). // ido & 2
		Raw("JZ single")
	p.comp12Body(false)
	p.comp12Advance(32)
	p.twNext(12)
	b.Raw("JMP single").
		Raw("firstpair:") // ido is 2 or 3: the first pair is the two-point group
	p.comp12Body(true)
	p.comp12Advance(32)
	p.twNext(12)
	b.Raw("single:").
		Raw("TESTQ $16, CX"). // ido & 1
		Raw("JZ knext")
	x.comp12Body(false)
	x.comp12Advance(16)
	// P advanced by S over the block; the next block starts 12·S after this
	// one. O advanced by S, which is the next block's output start.
	b.Raw("knext:").
		Raw("LEAQ (AX)(CX*8), AX").
		Raw("ADDQ R12, AX").
		Raw("DECQ %d(SP)", comp12Blocks).
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genComp12Last emits skLast12AVX2: the final radix-12 pass (ido == 1), two
// blocks per register as in genStockhamLastAVX2, an odd l1 finished by one
// 128-bit step.
func genComp12Last(f *emit.File) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("skLast12AVX2", sig, comp12Spill)
	y := skEmit{b: b, w: "Y", last: 12}
	x := skEmit{b: b, w: "X", last: 12}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("l1", "DX")
	b.Raw("MOVQ DX, R9").
		Raw("SHRQ $1, R9"). // block pairs
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("LEAQ (BX)(DX*8), R11").
		Raw("loop:").
		Raw("TESTQ R9, R9").
		Raw("JZ tail")
	y.comp12Body(false)
	b.Raw("ADDQ $%d, AX", 32*12)
	for _, r := range []string{"BX", "DI", "R11"} {
		b.Raw("ADDQ $32, %s", r)
	}
	b.Raw("DECQ R9").
		Raw("JMP loop").
		Raw("tail:").
		Raw("TESTQ $16, DX"). // l1 odd
		Raw("JZ done")
	x.comp12Body(false)
	b.Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// ---------------------------------------------------------------------------
// float32 real-FFT untangle and its inverse (Round 25), into
// untangle32_amd64.s.
//
// f32rUntangleAVX2(dst, z, tw *complex64, k *float32, m, quads int) is the
// fft package's f32Untangle for bins k = 1 .. 4·quads, and
// f32rRetangleAVX2(z, x, tw *complex64, k, h *float32, m, quads int) its
// f32Retangle loop (f32rRetangle) for the same k: the float64 untangle
// and retangle (smallUntangleStep) with single-precision instructions, four consecutive k per
// YMM register. The mirrored bins m-k-3 .. m-k are one 32-byte load whose
// four complex64 VPERMPD $0x1B reverses (and the same instruction reverses
// them back before the store), so lane q of both registers holds the pair
// (k+q, m-k-q). Every operation is the Go code's, in its order, separately
// rounded (gc does not fuse on GOAMD64=v1): the even/odd split as a blend
// (VBLENDPS) of the sum and the difference, the ±0.5 and the conjugations as
// sign flips (VXORPS with k's row 0, [0, -0, ...], the imaginary-lane mask),
// and W·xo as VMOVSLDUP/VMOVSHDUP and VADDSUBPS: (wr·xor − wi·xoi,
// wr·xoi + wi·xor), the Go products in the Go order. k is the float32
// constants table (sk32Fwd): row 0 the mask, row 1 0.5.
func genF32RealFile() {
	f := emit.NewFile("amd64")
	genF32rUntangleAVX2(f)
	genF32rRetangleAVX2(f)
	writeFile("untangle32_amd64.s", f.String())

	fb := emit.NewFile("amd64")
	for _, r := range []int{2, 3, 4, 5, 8} {
		genF32rBatchAVX2(fb, r)
	}
	writeFile("batch32_amd64.s", fb.String())
}

// genF32rBatchAVX2 emits sk32Batch{r}AVX2(cc, ch, tw *complex64, k *float32,
// ido, l1, quads, rem, jin, jout, adjin, adjout int) (Round 25): one float32
// Stockham pass of radix r over a batch of w = 4·quads + rem transforms laid
// side by side, the strips of a non-contiguous N-D axis (the fft package's
// skStage32.passBatch); genStockhamBatchAVX2 with single-precision registers.
// Point p of the batch is w neighbouring complex64 values, so four lines at
// one point are one YMM register (then two in an XMM register for rem&2 and
// one, VMOVSD, for rem&1), and every lane runs exactly the arithmetic of the
// 1-D float32 kernels (genF32StockhamAVX2) on one line: the same butterflies,
// the same twiddle product. A point's twiddle is the same for every line,
// broadcast with VBROADCASTSS from the batched table (for i = 1 .. ido-1, the
// r-1 twiddles of point i); point i = 0 is not multiplied, as in the Go pass.
//
// Registers as genF32StockhamAVX2's, with CX = jin = 8·ido·sIn (one input
// stream), DX = jout = 8·l1·ido·sOut; each row (one point) advances the
// streams by 8·w, then by adjin = 8·(sIn-w) and adjout = 8·(sOut-w).
func genF32rBatchAVX2(f *emit.File, r int) {
	names := []string{"cc", "ch", "tw", "k", "ido", "l1", "quads", "rem", "jin", "jout", "adjin", "adjout"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr}
	for range names[4:] {
		types = append(types, amd64.Int64)
	}
	sig := amd64.Layout(names, types, nil, nil)
	arg := func(name string) string {
		for _, p := range sig.Args {
			if p.Name == name {
				return fmt.Sprintf("%s+%d(FP)", name, p.Offset)
			}
		}
		panic(name)
	}
	b := amd64.NewFunc(fmt.Sprintf("sk32Batch%dAVX2", r), sig, 0)
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("tw", "R15").LoadArg("k", "R14").
		LoadArg("jin", "CX").LoadArg("jout", "DX").LoadArg("l1", "R8")
	b.Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("kloop:").
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Raw("LEAQ (AX)(CX*4), SI").
		Raw("LEAQ (BX)(DX*4), DI").
		Raw("MOVQ R15, R10")
	row := func(label string, notw bool) {
		y := f32simdEmit{b: b, w: "Y", notw: notw, bcast: !notw}
		x := f32simdEmit{b: b, w: "X", notw: notw, bcast: !notw}
		s := f32simdEmit{b: b, w: "S", notw: notw, bcast: !notw}
		b.Raw("MOVQ %s, R11", arg("quads")).
			Raw("%sq:", label).
			Raw("TESTQ R11, R11").
			Raw("JZ %sp", label)
		y.body(r, false)
		y.advance(32)
		b.Raw("DECQ R11").
			Raw("JMP %sq", label).
			Raw("%sp:", label).
			Raw("MOVQ %s, R11", arg("rem")).
			Raw("TESTQ $2, R11").
			Raw("JZ %ss", label)
		x.body(r, false)
		x.advance(16)
		b.Raw("%ss:", label).
			Raw("TESTQ $1, R11").
			Raw("JZ %se", label)
		s.body(r, false)
		s.advance(8)
		b.Raw("%se:", label).
			Raw("ADDQ %s, AX", arg("adjin")).
			Raw("ADDQ %s, SI", arg("adjin")).
			Raw("ADDQ %s, BX", arg("adjout")).
			Raw("ADDQ %s, DI", arg("adjout"))
	}
	row("zero", true) // point i = 0: no twiddle
	b.Raw("MOVQ %s, R9", arg("ido")).
		Raw("DECQ R9").
		Raw("iloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ knext")
	row("tw", false)
	b.Raw("ADDQ $%d, R10", (r-1)*8).
		Raw("DECQ R9").
		Raw("JMP iloop").
		Raw("knext:")
	f32simdNextBlock(b, r)
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// f32rMirrorSetup loads the four cursors of an untangle kernel: AX = &in[1],
// BX = &in[m-4], R9 = &out[1], R10 = &out[m-4], CX = &tw[1], with DX = m.
func f32rMirrorSetup(b *amd64.Builder, in, out string) {
	b.Raw("SHLQ $3, DX"). // m·8
				Raw("LEAQ 8(%s), AX", in).
				Raw("LEAQ -32(%s)(DX*1), BX", in).
				Raw("LEAQ 8(%s), R9", out).
				Raw("LEAQ -32(%s)(DX*1), R10", out).
				Raw("ADDQ $8, CX")
}

// f32rMirrorAdvance moves the cursors to the next four bins and loops.
func f32rMirrorAdvance(b *amd64.Builder) {
	b.Raw("ADDQ $32, AX").
		Raw("SUBQ $32, BX").
		Raw("ADDQ $32, R9").
		Raw("SUBQ $32, R10").
		Raw("ADDQ $32, CX").
		Raw("DECQ R8").
		Raw("JMP loop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
}

func genF32rUntangleAVX2(f *emit.File) {
	sig := amd64.Layout(
		[]string{"dst", "z", "tw", "k", "m", "quads"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("f32rUntangleAVX2", sig, 0)
	b.LoadArg("dst", "DI").LoadArg("z", "SI").LoadArg("tw", "CX").LoadArg("k", "R14").
		LoadArg("m", "DX").LoadArg("quads", "R8")
	f32rMirrorSetup(b, "SI", "DI")
	b.Raw("VMOVUPS 0(R14), Y12"). // imaginary-lane sign mask
					Raw("VMOVUPS 32(R14), Y13"). // 0.5
					Raw("loop:").
					Raw("TESTQ R8, R8").
					Raw("JZ done").
					Raw("VMOVUPS (AX), Y0").           // a = Z[k .. k+3]
					Raw("VMOVUPS (BX), Y1").           // Z[m-k-3 .. m-k]
					Raw("VPERMPD $0x1B, Y1, Y1").      // b = Z[m-k], Z[m-k-1], Z[m-k-2], Z[m-k-3]
					Raw("VADDPS Y1, Y0, Y2").          // s
					Raw("VSUBPS Y1, Y0, Y3").          // d
					Raw("VBLENDPS $0xAA, Y3, Y2, Y4"). // [s.re, d.im]
					Raw("VMULPS Y13, Y4, Y4").         // xe
					Raw("VBLENDPS $0x55, Y3, Y2, Y5"). // [d.re, s.im]
					Raw("VPERMILPS $0xB1, Y5, Y5").    // [s.im, d.re]
					Raw("VXORPS Y12, Y5, Y5").         // [s.im, -d.re]
					Raw("VMULPS Y13, Y5, Y5").         // xo
					Raw("VMOVSLDUP (CX), Y7").         // [wr, wr]
					Raw("VMOVSHDUP (CX), Y8").         // [wi, wi]
					Raw("VPERMILPS $0xB1, Y5, Y9").    // [xo.im, xo.re]
					Raw("VMULPS Y5, Y7, Y7").          // [wr·xo.re, wr·xo.im]
					Raw("VMULPS Y9, Y8, Y8").          // [wi·xo.im, wi·xo.re]
					Raw("VADDSUBPS Y8, Y7, Y7").       // t = w·xo
					Raw("VADDPS Y7, Y4, Y10").         // dst[k .. k+3]
					Raw("VSUBPS Y7, Y4, Y11").         // xe - t
					Raw("VXORPS Y12, Y11, Y11").       // conj: dst[m-k], ..., dst[m-k-3]
					Raw("VPERMPD $0x1B, Y11, Y11").    // dst[m-k-3 .. m-k]
					Raw("VMOVUPS Y10, (R9)").
					Raw("VMOVUPS Y11, (R10)")
	f32rMirrorAdvance(b)
	f.Add(b.Func())
}

func genF32rRetangleAVX2(f *emit.File) {
	sig := amd64.Layout(
		[]string{"z", "x", "tw", "k", "h", "m", "quads"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc("f32rRetangleAVX2", sig, 0)
	b.LoadArg("z", "DI").LoadArg("x", "SI").LoadArg("tw", "CX").LoadArg("k", "R14").
		LoadArg("h", "R15").LoadArg("m", "DX").LoadArg("quads", "R8")
	f32rMirrorSetup(b, "SI", "DI")
	b.Raw("VMOVUPS (R15), Y12"). // h
					Raw("VMOVUPS 0(R14), Y13"). // imaginary-lane sign mask
					Raw("loop:").
					Raw("TESTQ R8, R8").
					Raw("JZ done").
					Raw("VMOVUPS (AX), Y0"). // a = X[k .. k+3]
					Raw("VMOVUPS (BX), Y1").
					Raw("VPERMPD $0x1B, Y1, Y1").      // b = X[m-k], ..., X[m-k-3]
					Raw("VADDPS Y1, Y0, Y2").          // s
					Raw("VSUBPS Y1, Y0, Y3").          // d
					Raw("VBLENDPS $0xAA, Y3, Y2, Y4"). // [s.re, d.im]
					Raw("VMULPS Y12, Y4, Y4").         // xe
					Raw("VBLENDPS $0x55, Y3, Y2, Y5"). // [d.re, s.im]
					Raw("VMULPS Y12, Y5, Y5").         // dd = [dr, di]
					Raw("VMOVUPS (CX), Y6").
					Raw("VXORPS Y13, Y6, Y6").      // conj(w) = [wr, -wi]
					Raw("VMOVSLDUP Y6, Y7").        // [wr, wr]
					Raw("VMOVSHDUP Y6, Y8").        // [-wi, -wi]
					Raw("VPERMILPS $0xB1, Y5, Y9"). // [di, dr]
					Raw("VMULPS Y5, Y7, Y7").       // [wr·dr, wr·di]
					Raw("VMULPS Y9, Y8, Y8").       // [-wi·di, -wi·dr]
					Raw("VADDSUBPS Y8, Y7, Y7").    // xo = [wr·dr + wi·di, wr·di - wi·dr]
					Raw("VPERMILPS $0xB1, Y7, Y8"). // [xoi, xor]
					Raw("VXORPS Y13, Y8, Y8").      // [xoi, -xor]
					Raw("VSUBPS Y8, Y4, Y10").      // [xer - xoi, xei + xor] = Z[k]
					Raw("VADDPS Y8, Y4, Y11").      // [xer + xoi, xei - xor]
					Raw("VXORPS Y13, Y11, Y11").    // Z[m-k] = [xer + xoi, -(xei - xor)]
					Raw("VPERMPD $0x1B, Y11, Y11").
					Raw("VMOVUPS Y10, (R9)").
					Raw("VMOVUPS Y11, (R10)")
	f32rMirrorAdvance(b)
	f.Add(b.Func())
}

// ---------------------------------------------------------------------------
// Radix-10, -15 and -20 pass kernels (Round 26), into comp2_amd64.s: the fft
// package's radix5q.go, the prime-factor algorithm for r = 5·q, q = 2, 3, 4.
// Input n = (q·n1 + 5·n2) mod r, output k = (a·k1 + b·k2) mod r (a ≡ 1 mod 5,
// a ≡ 0 mod q, b ≡ 0 mod 5, b ≡ 1 mod q): q radix-5 butterflies (one per n2,
// over n1), then five radix-q butterflies (one per k1, over n2), with no
// twiddle in between. Every operation is the Go pass's (bfly5, then the add
// and subtract of radix 2, bfly3 or bfly4, then the outer twiddle product of
// twStore), separately rounded, so the kernels are bit-identical to it as the
// other AVX2 kernels are to theirs.
//
// Registers: the stage-1 values go to the frame (5·q slots of 32 bytes) and
// are read back for stage 2. Stream j sits 5·⌊j/5⌋ + j mod 5 strides from
// the start, through q bases: AX, SI, R8, R9 for the inputs (strides CX = S,
// R12 = 3S) and BX, DI, R11, R15 for the outputs (DX = OS, R13 = 3OS), each
// base reaching its five streams as +0, +1, +2, +3 (3S) and +4 (CX*4). R10
// is the twiddle cursor, R14 the constants; the blocks and the four-point
// groups left live in the frame.

// comp2Bases are the stream bases of the radix-5q kernels.
var (
	comp2InBase  = [4]string{"AX", "SI", "R8", "R9"}
	comp2OutBase = [4]string{"BX", "DI", "R11", "R15"}
)

// comp2Addr addresses stream j from its base: input streams j·S, output
// streams j·OS.
func comp2Addr(out bool, j int) string {
	base, s, s3 := comp2InBase[j/5], "CX", "R12"
	if out {
		base, s, s3 = comp2OutBase[j/5], "DX", "R13"
	}
	switch j % 5 {
	case 0:
		return "(" + base + ")"
	case 1:
		return fmt.Sprintf("(%s)(%s*1)", base, s)
	case 2:
		return fmt.Sprintf("(%s)(%s*2)", base, s)
	case 3:
		return fmt.Sprintf("(%s)(%s*1)", base, s3)
	}
	return fmt.Sprintf("(%s)(%s*4)", base, s)
}

// comp2LastAddr addresses output stream j of a final-pass kernel (the inputs
// are read through AX by skEmit.in).
func comp2LastAddr(out bool, j int) string {
	if !out {
		panic("comp2LastAddr: the final pass reads through skEmit.in")
	}
	return comp2Addr(true, j)
}

// comp2Map is radix5q.go's index map for r = 5·q.
type comp2Map struct {
	q   int
	in  [4][5]int
	out [5][4]int
}

func comp2NewMap(q int) comp2Map {
	r := 5 * q
	var a, b int
	for x := range r {
		if x%5 == 1 && x%q == 0 {
			a = x
		}
		if x%5 == 0 && x%q == 1 {
			b = x
		}
	}
	m := comp2Map{q: q}
	for n2 := range q {
		for n1 := range 5 {
			m.in[n2][n1] = (q*n1 + 5*n2) % r
		}
	}
	for k1 := range 5 {
		for k2 := range q {
			m.out[k1][k2] = (a*k1 + b*k2) % r
		}
	}
	return m
}

// comp2Slot is the frame offset of stage-1 value (n2, k1).
func comp2Slot(q, n2, k1 int) string { return fmt.Sprintf("%d(SP)", 32*(q*k1+n2)) }

// comp2Bfly5 emits the Go bfly5 on input streams js (the body of genStockham's
// radix 5, its loads included) and leaves y0..y4 in Y7, Y1, Y2, Y3, Y4.
func (e skEmit) comp2Bfly5(js [5]int) [5]int {
	e.in(0, js[0])
	e.in(1, js[1])
	e.in(2, js[4])
	e.add(3, 1, 2) // t1
	e.sub(4, 1, 2) // t2
	e.in(1, js[2])
	e.in(2, js[3])
	e.add(5, 1, 2) // t3
	e.sub(6, 1, 2) // t4
	e.add(7, 0, 3)
	e.add(7, 7, 5) // y0
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
	return [5]int{7, 1, 2, 3, 4}
}

// comp2Body emits one radix-5q butterfly over the points the register width
// holds, with the outer twiddles (twStore) unless this is a final pass.
func (e skEmit) comp2Body(m comp2Map, first bool) {
	q := m.q
	for n2 := range q {
		y := e.comp2Bfly5(m.in[n2])
		for k1, r := range y {
			e.raw("VMOVUPD %s, %s", e.v(r), comp2Slot(q, n2, k1))
		}
	}
	for k1 := range 5 {
		for n2 := range q {
			e.raw("VMOVUPD %s, %s", comp2Slot(q, n2, k1), e.v(n2))
		}
		var z []int
		switch q {
		case 2:
			e.add(2, 0, 1)
			e.sub(3, 0, 1)
			z = []int{2, 3}
		case 3:
			e.comp12Bfly3([3]int{0, 1, 2}, [3]int{3, 4, 5}, [3]int{6, 7, 8})
			z = []int{3, 4, 5}
		default:
			e.bfly4r([4]int{0, 1, 2, 3}, [4]int{4, 5, 6, 7}, [4]int{8, 9, 10, 11})
			z = []int{8, 9, 10, 11}
		}
		for k2, r := range z {
			if j := m.out[k1][k2]; j == 0 {
				e.st(r, e.outAddr(0))
			} else {
				e.twStore(r, j, 12, 13, 14, 15, first)
			}
		}
	}
}

// comp2Advance moves the stream bases of a radix-5q pass kernel.
func comp2Advance(b *amd64.Builder, q, bytes int) {
	for _, r := range comp2InBase[:q] {
		b.Raw("ADDQ $%d, %s", bytes, r)
	}
	for _, r := range comp2OutBase[:q] {
		b.Raw("ADDQ $%d, %s", bytes, r)
	}
}

// genComp2 emits skPass{r}AVX2 for r = 5·q: the loop of genStockham16 around
// comp2Body.
func genComp2(f *emit.File, q int) {
	r := 5 * q
	m := comp2NewMap(q)
	spill := 32 * r
	blocks, groups := spill, spill+8
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skPass%dAVX2", r), sig, spill+16)
	q0 := skEmit{b: b, w: "Y", g: 4, addr: comp2Addr}
	q1 := skEmit{b: b, w: "Y", g: 4, half: 1, addr: comp2Addr}
	p := skEmit{b: b, w: "Y", g: 2, addr: comp2Addr}
	x := skEmit{b: b, w: "X", g: 1, addr: comp2Addr}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").
		LoadArg("ido", "CX").LoadArg("l1", "R8")
	b.Raw("MOVQ CX, DX").
		Raw("IMULQ R8, DX").
		Raw("SHLQ $4, CX"). // S
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (CX)(CX*2), R12").
		Raw("LEAQ (DX)(DX*2), R13").
		Raw("MOVQ R8, %d(SP)", blocks).
		Raw("kloop:").
		Raw("CMPQ %d(SP), $0", blocks).
		Raw("JEQ done").
		Raw("MOVQ CX, R9").
		Raw("SHRQ $6, R9"). // four-point groups = ido/4
		Raw("MOVQ R9, %d(SP)", groups)
	// Base t+1 is base t plus five strides.
	for t := 1; t < q; t++ {
		b.Raw("LEAQ (%s)(CX*4), %s", comp2InBase[t-1], comp2InBase[t]).
			Raw("ADDQ CX, %s", comp2InBase[t]).
			Raw("LEAQ (%s)(DX*4), %s", comp2OutBase[t-1], comp2OutBase[t]).
			Raw("ADDQ DX, %s", comp2OutBase[t])
	}
	b.LoadArg("tw", "R10")
	b.Raw("CMPQ %d(SP), $0", groups).
		Raw("JEQ firstpair")
	q0.comp2Body(m, true)
	comp2Advance(b, q, 32)
	q1.comp2Body(m, false)
	comp2Advance(b, q, 32)
	q0.twNext(r)
	b.Raw("DECQ %d(SP)", groups).
		Raw("qloop:").
		Raw("CMPQ %d(SP), $0", groups).
		Raw("JEQ pair")
	q0.comp2Body(m, false)
	comp2Advance(b, q, 32)
	q1.comp2Body(m, false)
	comp2Advance(b, q, 32)
	q0.twNext(r)
	b.Raw("DECQ %d(SP)", groups).
		Raw("JMP qloop").
		Raw("pair:").
		Raw("TESTQ $32, CX"). // ido & 2
		Raw("JZ single")
	p.comp2Body(m, false)
	comp2Advance(b, q, 32)
	p.twNext(r)
	b.Raw("JMP single").
		Raw("firstpair:") // ido is 2 or 3: the first pair is the two-point group
	p.comp2Body(m, true)
	comp2Advance(b, q, 32)
	p.twNext(r)
	b.Raw("single:").
		Raw("TESTQ $16, CX"). // ido & 1
		Raw("JZ knext")
	x.comp2Body(m, false)
	comp2Advance(b, q, 16)
	// The last base advanced by S over the block, from 5·(q-1)·S past the
	// block's start; the next block starts r·S after it, four strides on.
	// O advanced by S, which is the next block's output start.
	b.Raw("knext:").
		Raw("LEAQ (%s)(CX*4), AX", comp2InBase[q-1]).
		Raw("DECQ %d(SP)", blocks).
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genComp2Last emits skLast{r}AVX2 for r = 5·q: the final pass (ido == 1),
// two blocks per register as in genStockhamLastAVX2, an odd l1 finished by
// one 128-bit step.
func genComp2Last(f *emit.File, q int) {
	r := 5 * q
	m := comp2NewMap(q)
	sig := amd64.Layout(
		[]string{"cc", "ch", "k", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64}, nil, nil,
	)
	b := amd64.NewFunc(fmt.Sprintf("skLast%dAVX2", r), sig, 32*r)
	y := skEmit{b: b, w: "Y", last: r, addr: comp2LastAddr}
	x := skEmit{b: b, w: "X", last: r, addr: comp2LastAddr}
	b.LoadArg("cc", "AX").LoadArg("ch", "BX").LoadArg("k", "R14").LoadArg("l1", "DX")
	b.Raw("MOVQ DX, R9").
		Raw("SHRQ $1, R9"). // block pairs
		Raw("SHLQ $4, DX"). // OS
		Raw("LEAQ (DX)(DX*2), R13")
	for t := 1; t < q; t++ {
		b.Raw("LEAQ (%s)(DX*4), %s", comp2OutBase[t-1], comp2OutBase[t]).
			Raw("ADDQ DX, %s", comp2OutBase[t])
	}
	b.Raw("loop:").
		Raw("TESTQ R9, R9").
		Raw("JZ tail")
	y.comp2Body(m, false)
	b.Raw("ADDQ $%d, AX", 32*r)
	for _, o := range comp2OutBase[:q] {
		b.Raw("ADDQ $32, %s", o)
	}
	b.Raw("DECQ R9").
		Raw("JMP loop").
		Raw("tail:").
		Raw("TESTQ $16, DX"). // l1 odd
		Raw("JZ done")
	x.comp2Body(m, false)
	b.Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genComp2File writes comp2_amd64.s: the radix-10, -15 and -20 kernels.
func genComp2File() {
	f := emit.NewFile("amd64")
	for _, q := range []int{2, 3, 4} {
		genComp2(f, q)
		genComp2Last(f, q)
	}
	writeFile("comp2_amd64.s", f.String())
}

// ---------------------------------------------------------------------------
// The split layout at 512 bits (Round 28, AVX-512 Intel). genStockhamSplit's
// kernels with eight points per register: eight points 8q .. 8q+7 occupy the
// 128 bytes they occupy interleaved, as their real parts in the order (0, 4,
// 1, 5, 2, 6, 3, 7), then their imaginary parts in the same order. That is
// what VUNPCKLPD and VUNPCKHPD make of two interleaved ZMM loads
// ([r0 i0 r1 i1 r2 i2 r3 i3], [r4 i4 … r7 i7] → [r0 r4 r1 r5 r2 r6 r3 r7]),
// one in-lane shuffle per register each way, as at 256 bits; the twiddles are
// stored in that order (kernels.intelSplitTwiddles512). The bodies are
// genStockhamSplit's, operation for operation, so the result is the scalar
// pass's bits. The i = 0 point (lane 0 of a block's first group) is put
// back by a blend under opmask K1 = 1. The two constants (√2/2 and −0,
// kernels.splitK's rows) are broadcast from memory by the instructions that
// use them; the sign flip is VPXORQ, which AVX-512F has (VXORPD on ZMM needs
// AVX512DQ). Only Z0..Z15 are written, and the radix-8 body spills four of
// them to its frame as genStockhamSplit's does. VZEROUPPER does not clean
// Z16..Z31: a first build that kept those four values and the constants
// there ran the powers of two 1.07–1.08× faster at 2048 and 4096, but every
// composite transform run after it in the same process 4–5% slower on
// Cascade Lake (Round 28), and zeroing the registers on exit did not undo it.
//
// intelSplit512R{r}{in}{out}{Fwd|Inv}(cc, ch, tw *complex128, k *float64,
// ido, l1 int), ido a positive multiple of eight. Registers as in
// genStockhamSplit.

type intelSplit512Emit struct {
	b         *amd64.Builder
	r         int
	inS, outS bool
	inverse   bool
}

func (e intelSplit512Emit) raw(format string, a ...any) { e.b.Raw(format, a...) }

// intelZ names a ZMM register.
func intelZ(i int) string { return fmt.Sprintf("Z%d", i) }

// intelAt64 is addr 64 bytes further on.
func intelAt64(addr string) string { return "64" + addr }

// intelSplit512Frame holds the radix-8 body's four spilled registers.
const intelSplit512Frame = 256

func (e intelSplit512Emit) load(j, re, im, t0, t1 int) {
	a := skIn(j)
	if e.inS {
		e.raw("VMOVUPD %s, %s", a, intelZ(re))
		e.raw("VMOVUPD %s, %s", intelAt64(a), intelZ(im))
		return
	}
	e.raw("VMOVUPD %s, %s", a, intelZ(t0))
	e.raw("VMOVUPD %s, %s", intelAt64(a), intelZ(t1))
	e.raw("VUNPCKLPD %s, %s, %s", intelZ(t1), intelZ(t0), intelZ(re))
	e.raw("VUNPCKHPD %s, %s, %s", intelZ(t1), intelZ(t0), intelZ(im))
}

func (e intelSplit512Emit) store(j, re, im, t0, t1 int) {
	a := skOut(j)
	if e.outS {
		e.raw("VMOVUPD %s, %s", intelZ(re), a)
		e.raw("VMOVUPD %s, %s", intelZ(im), intelAt64(a))
		return
	}
	e.raw("VUNPCKLPD %s, %s, %s", intelZ(im), intelZ(re), intelZ(t0))
	e.raw("VUNPCKHPD %s, %s, %s", intelZ(im), intelZ(re), intelZ(t1))
	e.raw("VMOVUPD %s, %s", intelZ(t0), a)
	e.raw("VMOVUPD %s, %s", intelZ(t1), intelAt64(a))
}

func (e intelSplit512Emit) add(dst, a, b int) {
	e.raw("VADDPD %s, %s, %s", intelZ(b), intelZ(a), intelZ(dst))
}

func (e intelSplit512Emit) sub(dst, a, b int) {
	e.raw("VSUBPD %s, %s, %s", intelZ(b), intelZ(a), intelZ(dst))
}

// twStore is splitEmit.twStore at 512 bits: twiddle j of the group is 128
// bytes, eight real parts then eight imaginary parts.
func (e intelSplit512Emit) twStore(re, im, j int, t [4]int, first bool) {
	off := (j - 1) * 128
	e.raw("VMOVUPD %d(R10), %s", off, intelZ(t[0]))    // wr
	e.raw("VMOVUPD %d(R10), %s", off+64, intelZ(t[1])) // wi
	e.raw("VMULPD %s, %s, %s", intelZ(t[0]), intelZ(re), intelZ(t[2]))
	e.raw("VMULPD %s, %s, %s", intelZ(t[1]), intelZ(im), intelZ(t[3]))
	e.sub(t[2], t[2], t[3]) // yr·wr − yi·wi
	e.raw("VMULPD %s, %s, %s", intelZ(t[0]), intelZ(im), intelZ(t[0]))
	e.raw("VMULPD %s, %s, %s", intelZ(t[1]), intelZ(re), intelZ(t[1]))
	e.add(t[0], t[0], t[1]) // yi·wr + yr·wi
	if first {
		// K1 = lane 0: point i = 0 keeps the unmultiplied value.
		e.raw("VBLENDMPD %s, %s, K1, %s", intelZ(re), intelZ(t[2]), intelZ(t[2]))
		e.raw("VBLENDMPD %s, %s, K1, %s", intelZ(im), intelZ(t[0]), intelZ(t[0]))
	}
	e.store(j, t[2], t[0], t[1], t[3])
}

// rotAdd is splitEmit.rotAdd: pr, pi = a + rotS(b); mr, mi = a − rotS(b).
func (e intelSplit512Emit) rotAdd(pr, pi, mr, mi, ar, ai, br, bi int) {
	if !e.inverse {
		e.add(pr, ar, bi)
		e.sub(pi, ai, br)
		e.sub(mr, ar, bi)
		e.add(mi, ai, br)
		return
	}
	e.sub(pr, ar, bi)
	e.add(pi, ai, br)
	e.add(mr, ar, bi)
	e.sub(mi, ai, br)
}

// body4 is splitEmit.body's radix-4 case.
func (e intelSplit512Emit) body4(first bool) {
	for j := 0; j < 4; j++ {
		e.load(j, 2*j, 2*j+1, 12, 13)
	}
	e.add(8, 0, 4) // t2 = x0 + x2
	e.sub(0, 0, 4) // t1 = x0 − x2
	e.add(9, 1, 5)
	e.sub(1, 1, 5)
	e.add(4, 2, 6) // t3 = x1 + x3
	e.sub(2, 2, 6) // t4 = x1 − x3
	e.add(5, 3, 7)
	e.sub(3, 3, 7)
	e.add(6, 8, 4) // y0 = t2 + t3
	e.add(7, 9, 5)
	e.store(0, 6, 7, 10, 11)
	e.sub(8, 8, 4) // y2 = t2 − t3
	e.sub(9, 9, 5)
	e.rotAdd(4, 5, 0, 1, 0, 1, 2, 3) // y1, y3 = t1 ± rotS(t4)
	t := [4]int{10, 11, 12, 13}
	e.twStore(4, 5, 1, t, first)
	e.twStore(8, 9, 2, t, first)
	e.twStore(0, 1, 3, t, first)
}

// neg flips a register's signs in place: −x, exactly (VPXORQ with −0
// broadcast from the constants).
func (e intelSplit512Emit) neg(r int) {
	e.raw("VPXORQ.BCST %d(R14), %s, %s", splitKNeg, intelZ(r), intelZ(r))
}

// body8 is splitEmit.body8 at 512 bits, spilling the same four registers.
func (e intelSplit512Emit) body8(first bool) {
	// Odd half.
	e.load(1, 0, 1, 12, 13)
	e.load(5, 2, 3, 12, 13)
	e.add(4, 0, 2) // a1 = x1 + x5
	e.sub(0, 0, 2) // a5 = x1 − x5
	e.add(2, 1, 3)
	e.sub(1, 1, 3) // a1 (4,2), a5 (0,1)
	e.load(3, 3, 5, 12, 13)
	e.load(7, 6, 7, 12, 13)
	e.add(8, 3, 6) // a3 = x3 + x7
	e.sub(3, 3, 6) // a7 = x3 − x7
	e.add(6, 5, 7)
	e.sub(5, 5, 7) // a3 (8,6), a7 (3,5)
	e.add(7, 4, 8) // a1 + a3
	e.sub(4, 4, 8) // a1 − a3
	e.add(8, 2, 6)
	e.sub(2, 2, 6) // a1 (7,8), a3 (4,2), before rotS(a3)
	e.raw("VMOVUPD %s, 0(SP)", intelZ(7))
	e.raw("VMOVUPD %s, 64(SP)", intelZ(8))
	e.raw("VMOVUPD %s, 128(SP)", intelZ(4))
	e.raw("VMOVUPD %s, 192(SP)", intelZ(2))
	e.rotAdd(6, 9, 0, 1, 0, 1, 3, 5) // a5, a7 = a5 ± rotS(a7): a5 (6,9), a7 (0,1)
	if !e.inverse {
		// a5 = h·(a5r + a5i, a5i − a5r); a7 = h·(a7i − a7r, −a7r − a7i).
		e.add(3, 6, 9)
		e.sub(5, 9, 6)
		e.sub(6, 1, 0)
		e.neg(0)
		e.sub(0, 0, 1)
	} else {
		// a5 = h·(a5r − a5i, a5i + a5r); a7 = h·(−a7i − a7r, a7r − a7i).
		e.sub(3, 6, 9)
		e.add(5, 9, 6)
		e.sub(2, 0, 1)
		e.neg(1)
		e.sub(6, 1, 0)
		e.raw("VMOVAPD %s, %s", intelZ(2), intelZ(0))
	}
	for _, r := range []int{3, 5, 6, 0} {
		e.raw("VMULPD.BCST %d(R14), %s, %s", splitKH, intelZ(r), intelZ(r))
	} // a5 (3,5), a7 (6,0)
	// Even half.
	e.load(0, 1, 2, 13, 14)
	e.load(4, 4, 7, 13, 14)
	e.add(8, 1, 4) // a0 = x0 + x4
	e.sub(1, 1, 4) // a4 = x0 − x4
	e.add(4, 2, 7)
	e.sub(2, 2, 7) // a0 (8,4), a4 (1,2)
	e.load(2, 7, 9, 13, 14)
	e.load(6, 10, 11, 13, 14)
	e.add(12, 7, 10) // a2 = x2 + x6
	e.sub(7, 7, 10)  // a6 = x2 − x6
	e.add(10, 9, 11)
	e.sub(9, 9, 11) // a2 (12,10), a6 (7,9)
	e.add(11, 8, 12)
	e.sub(8, 8, 12)
	e.add(12, 4, 10)
	e.sub(4, 4, 10)                    // a0 (11,12), a2 (8,4)
	e.rotAdd(10, 13, 1, 2, 1, 2, 7, 9) // a4, a6 = a4 ± rotS(a6): a4 (10,13), a6 (1,2)
	t := [4]int{3, 5, 14, 15}
	// y1, y5 = a4 ± a5.
	e.add(7, 10, 3)
	e.sub(10, 10, 3)
	e.add(9, 13, 5)
	e.sub(13, 13, 5)
	e.twStore(7, 9, 1, t, first)
	e.twStore(10, 13, 5, t, first)
	// y3, y7 = a6 ± a7.
	t = [4]int{3, 5, 10, 13}
	e.add(7, 1, 6)
	e.sub(1, 1, 6)
	e.add(9, 2, 0)
	e.sub(2, 2, 0)
	e.twStore(7, 9, 3, t, first)
	e.twStore(1, 2, 7, t, first)
	// y0, y4 = a0 ± a1.
	t = [4]int{0, 1, 2, 3}
	e.raw("VADDPD 0(SP), %s, %s", intelZ(11), intelZ(7))
	e.raw("VSUBPD 0(SP), %s, %s", intelZ(11), intelZ(11))
	e.raw("VADDPD 64(SP), %s, %s", intelZ(12), intelZ(9))
	e.raw("VSUBPD 64(SP), %s, %s", intelZ(12), intelZ(12))
	e.store(0, 7, 9, 5, 6)
	e.twStore(11, 12, 4, t, first)
	// y2, y6 = a2 ± rotS(a3).
	e.raw("VMOVUPD 128(SP), %s", intelZ(6))
	e.raw("VMOVUPD 192(SP), %s", intelZ(0))
	e.rotAdd(7, 9, 8, 4, 8, 4, 6, 0)
	t = [4]int{1, 2, 3, 5}
	e.twStore(7, 9, 2, t, first)
	e.twStore(8, 4, 6, t, first)
}

func (e intelSplit512Emit) body(first bool) {
	if e.r == 4 {
		e.body4(first)
		return
	}
	e.body8(first)
}

func (e intelSplit512Emit) advance(bytes int) {
	regs := []string{"AX", "BX"}
	if e.r > 4 {
		regs = append(regs, "SI", "DI")
	}
	for _, r := range regs {
		e.raw("ADDQ $%d, %s", bytes, r)
	}
}

// intelSplit512Name is the kernel's symbol.
func intelSplit512Name(r int, inS, outS, inverse bool) string {
	io := map[bool]string{false: "I", true: "S"}
	dir := "Fwd"
	if inverse {
		dir = "Inv"
	}
	return fmt.Sprintf("intelSplit512R%d%s%s%s", r, io[inS], io[outS], dir)
}

// genIntelSplit512 emits one 512-bit split-layout pass kernel (see above).
func genIntelSplit512(f *emit.File, r int, inS, outS, inverse bool) {
	sig := amd64.Layout(
		[]string{"cc", "ch", "tw", "k", "ido", "l1"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}, nil, nil,
	)
	frame := 0
	if r == 8 {
		frame = intelSplit512Frame
	}
	b := amd64.NewFunc(intelSplit512Name(r, inS, outS, inverse), sig, frame)
	e := intelSplit512Emit{b: b, r: r, inS: inS, outS: outS, inverse: inverse}
	b.Raw("MOVQ $1, R9").Raw("KMOVB R9, K1") // lane 0: the i = 0 point
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
		Raw("JZ done")
	if r > 4 {
		b.Raw("LEAQ (AX)(CX*4), SI").
			Raw("LEAQ (BX)(DX*4), DI")
	}
	b.Raw("MOVQ R15, R10").
		Raw("MOVQ CX, R9").
		Raw("SHRQ $7, R9") // groups = ido/8
	e.body(true)
	e.advance(128)
	b.Raw("ADDQ $%d, R10", (r-1)*128).
		Raw("DECQ R9").
		Raw("gloop:").
		Raw("TESTQ R9, R9").
		Raw("JZ knext")
	e.body(false)
	e.advance(128)
	b.Raw("ADDQ $%d, R10", (r-1)*128).
		Raw("DECQ R9").
		Raw("JMP gloop").
		Raw("knext:")
	skNextBlock(b, r)
	b.Raw("DECQ R8").
		Raw("JMP kloop").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// genIntelSplit512File writes the 512-bit split-layout pass kernels: radix 4
// and 8, every pair of input and output layouts, both directions.
func genIntelSplit512File() {
	f := emit.NewFile("amd64")
	for _, r := range []int{4, 8} {
		for _, inverse := range []bool{false, true} {
			for _, io := range [][2]bool{{false, true}, {true, true}, {true, false}, {false, false}} {
				genIntelSplit512(f, r, io[0], io[1], inverse)
			}
		}
	}
	writeFile("stockhamsplit512_amd64.s", f.String())
}

// ---------------------------------------------------------------------------
// The float64 real-FFT untangle and its inverse (Round 4; Round 30), into
// untangle_amd64.s.
//
// smallUntangleAVX2(dst, z, tw *complex128, k *skConst, m, pairs int) is the
// fft package's rfftUntangle for bins k = 1 .. 2·pairs, two consecutive k per
// YMM register. Bin k reads Z[k] and Z[m-k]; for k and k+1 those are one
// forward load of Z[k..k+1] and one load of Z[m-k-1..m-k] with its 128-bit
// halves swapped, and the two mirrored outputs dst[m-k], dst[m-k-1] are stored
// the same way. Per bin, with Z[k] = a and Z[m-k] = b, s = a + b and
// d = a - b:
//
//	xe = [s.re, d.im]·0.5          (a blend, then the multiply)
//	xo = [s.im, -d.re]·0.5         (the other blend, a sign flip)
//	t  = W^k · xo                  (the complex product, VADDSUBPD)
//	dst[k] = xe + t;  dst[m-k] = conj(xe - t)
//
// which is rfftUntangle's arithmetic operation for operation, separately
// rounded with no FMA: bit-identical at GOAMD64=v1. k's row 0 (the forward
// rotation mask, [0, -0, 0, -0]) is the imaginary-lane sign flip, row 1 is
// 0.5. smallRetangleAVX2(z, x, tw *complex128, k *skConst, h *[4]float64, m,
// pairs int) is irfftRetangle's loop the same way: with X[k] = a, X[m-k] = b
// and h = 0.5·scale,
//
//	xe = [s.re, d.im]·h;  dd = [d.re, s.im]·h
//	xo = conj(W^k)·dd
//	Z[k] = xe + i·xo;  Z[m-k] = conj(xe - i·xo)
//
// Round 4's kernels ran one step (two bins and their mirrors) per iteration,
// a chain of some twenty dependent instructions, with five shuffles besides
// the two lane swaps. Round 30 rebuilt them two ways, both of which pay on
// Zen 3: three independent steps per iteration, their instructions
// interleaved one by one (Round 27's lesson on Neoverse-N1), each step in its
// own four registers; and two shuffles fewer per step: the twiddle's real
// and imaginary parts come duplicated straight from memory (VMOVDDUP at tw
// and at tw+8, no VPERMILPD), and the untangle builds xo with its lanes
// swapped, [xo.im, xo.re] = [-d.re, s.im]·0.5 (the sign flipped before the
// halving, which is exact either way), so that one swap serves both
// products. Every product, sum and difference keeps the operands, in the
// order, of the Go code, so the results are the same bits. The pairs left
// over run one step at a time.

// genSmallUntangleFile writes untangle_amd64.s.
func genSmallUntangleFile() {
	f := emit.NewFile("amd64")
	genSmallUntangle(f, false)
	genSmallUntangle(f, true)
	genClUntangle512(f, false)
	genClUntangle512(f, true)
	writeFile("untangle_amd64.s", f.String())
}

// smallUntangleSteps is how many steps one iteration interleaves: three use
// twelve registers, the constants three more.
const smallUntangleSteps = 3

// smallUntangleRegs gives step j its four registers Y(4j) .. Y(4j+3).
func smallUntangleRegs(j int) (a, b, c, d string) {
	r := func(i int) string { return fmt.Sprintf("Y%d", 4*j+i) }
	return r(0), r(1), r(2), r(3)
}

// smallUntangleStep returns step j's instructions, its loads and stores at
// offset 32·j from the cursors (AX forward input, BX mirrored input, CX
// twiddles, R9 forward output, R10 mirrored output). Y15 holds the
// imaginary-lane sign mask, Y13 the real-lane one, Y12 both lanes'; Y14
// holds 0.5 (untangle) or h (retangle).
func smallUntangleStep(j int, inverse bool) []string {
	a, b, c, d := smallUntangleRegs(j)
	o := 32 * j
	in := []string{
		fmt.Sprintf("VMOVUPD %d(AX), %s", o, a),
		fmt.Sprintf("VMOVUPD %d(BX), %s", -o, b),
		fmt.Sprintf("VPERM2F128 $1, %s, %s, %s", b, b, b),
		fmt.Sprintf("VADDPD %s, %s, %s", b, a, c), // s
		fmt.Sprintf("VSUBPD %s, %s, %s", b, a, d), // d
		fmt.Sprintf("VBLENDPD $10, %s, %s, %s", d, c, a),
		fmt.Sprintf("VMULPD Y14, %s, %s", a, a), // xe
		fmt.Sprintf("VBLENDPD $5, %s, %s, %s", d, c, b),
	}
	if !inverse {
		return append(in,
			fmt.Sprintf("VXORPD Y13, %s, %s", b, b),
			fmt.Sprintf("VMULPD Y14, %s, %s", b, b),    // [xo.im, xo.re]
			fmt.Sprintf("VMOVDDUP %d(CX), %s", o, c),   // [wr, wr]
			fmt.Sprintf("VMOVDDUP %d(CX), %s", o+8, d), // [wi, wi]
			fmt.Sprintf("VMULPD %s, %s, %s", b, d, d),
			fmt.Sprintf("VPERMILPD $5, %s, %s", b, b), // xo
			fmt.Sprintf("VMULPD %s, %s, %s", b, c, c),
			fmt.Sprintf("VADDSUBPD %s, %s, %s", d, c, c), // t
			fmt.Sprintf("VADDPD %s, %s, %s", c, a, b),
			fmt.Sprintf("VSUBPD %s, %s, %s", c, a, d),
			fmt.Sprintf("VXORPD Y15, %s, %s", d, d),
			fmt.Sprintf("VPERM2F128 $1, %s, %s, %s", d, d, d),
			fmt.Sprintf("VMOVUPD %s, %d(R9)", b, o),
			fmt.Sprintf("VMOVUPD %s, %d(R10)", d, -o),
		)
	}
	return append(in,
		fmt.Sprintf("VMULPD Y14, %s, %s", b, b),    // dd
		fmt.Sprintf("VMOVDDUP %d(CX), %s", o, d),   // [wr, wr]
		fmt.Sprintf("VMOVDDUP %d(CX), %s", o+8, c), // [wi, wi]
		fmt.Sprintf("VXORPD Y12, %s, %s", c, c),    // [-wi, -wi]
		fmt.Sprintf("VMULPD %s, %s, %s", b, d, d),
		fmt.Sprintf("VPERMILPD $5, %s, %s", b, b),
		fmt.Sprintf("VMULPD %s, %s, %s", b, c, c),
		fmt.Sprintf("VADDSUBPD %s, %s, %s", c, d, d), // xo
		fmt.Sprintf("VPERMILPD $5, %s, %s", d, b),
		fmt.Sprintf("VXORPD Y15, %s, %s", b, b), // [xoi, -xor]
		fmt.Sprintf("VSUBPD %s, %s, %s", b, a, c),
		fmt.Sprintf("VADDPD %s, %s, %s", b, a, d),
		fmt.Sprintf("VXORPD Y15, %s, %s", d, d),
		fmt.Sprintf("VPERM2F128 $1, %s, %s, %s", d, d, d),
		fmt.Sprintf("VMOVUPD %s, %d(R9)", c, o),
		fmt.Sprintf("VMOVUPD %s, %d(R10)", d, -o),
	)
}

// smallAdvance moves the five cursors past n steps.
func smallAdvance(b *amd64.Builder, n int) {
	b.Raw("ADDQ $%d, AX", 32*n).
		Raw("SUBQ $%d, BX", 32*n).
		Raw("ADDQ $%d, R9", 32*n).
		Raw("SUBQ $%d, R10", 32*n).
		Raw("ADDQ $%d, CX", 32*n)
}

// genSmallUntangle emits smallUntangleAVX2 or, when inverse,
// smallRetangleAVX2: smallUntangleSteps steps per iteration while that many
// pairs are left, then one at a time. The last step reads tw[2·pairs+1].re
// (the VMOVDDUP at tw+8), which the Go wrappers bound-check.
func genSmallUntangle(f *emit.File, inverse bool) {
	name := "smallUntangleAVX2"
	names := []string{"dst", "z", "tw", "k", "m", "pairs"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}
	if inverse {
		name = "smallRetangleAVX2"
		names = []string{"z", "x", "tw", "k", "h", "m", "pairs"}
		types = []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}
	}
	b := amd64.NewFunc(name, amd64.Layout(names, types, nil, nil), 0)
	b.LoadArg(names[0], "DI").LoadArg(names[1], "SI").LoadArg("tw", "CX").LoadArg("k", "R14").
		LoadArg("m", "DX").LoadArg("pairs", "R8")
	b.Raw("SHLQ $4, DX").
		Raw("LEAQ 16(SI), AX").         // &in[1]
		Raw("LEAQ -32(SI)(DX*1), BX").  // &in[m-2]
		Raw("LEAQ 16(DI), R9").         // &out[1]
		Raw("LEAQ -32(DI)(DX*1), R10"). // &out[m-2]
		Raw("ADDQ $16, CX").            // &tw[1]
		Raw("VMOVUPD 0(R14), Y15").     // imaginary-lane sign mask [0, -0]
		Raw("VPERMILPD $5, Y15, Y13").  // real-lane sign mask [-0, 0]
		Raw("VORPD Y15, Y13, Y12")      // both lanes
	if inverse {
		b.LoadArg("h", "R15").Raw("VMOVUPD (R15), Y14")
	} else {
		b.Raw("VMOVUPD 32(R14), Y14") // 0.5
	}
	b.Raw("loop:").
		Raw("CMPQ R8, $%d", smallUntangleSteps).
		Raw("JLT tail")
	seqs := make([][]string, smallUntangleSteps)
	for j := range seqs {
		seqs[j] = smallUntangleStep(j, inverse)
	}
	for i := range seqs[0] {
		for j := range seqs {
			b.Raw("%s", seqs[j][i])
		}
	}
	smallAdvance(b, smallUntangleSteps)
	b.Raw("SUBQ $%d, R8", smallUntangleSteps).
		Raw("JMP loop").
		Raw("tail:").
		Raw("TESTQ R8, R8").
		Raw("JZ done")
	for _, ins := range smallUntangleStep(0, inverse) {
		b.Raw("%s", ins)
	}
	smallAdvance(b, 1)
	b.Raw("DECQ R8").
		Raw("JMP tail").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	f.Add(b.Func())
}

// clUntangleStep512 is smallUntangleStep at 512 bits (Round 31, Cascade
// Lake): four bins k .. k+3 and their mirrors m-k-3 .. m-k per step, at
// offset 64·j, in Z(4j) .. Z(4j+3). The operations are the AVX2 step's, lane
// for lane, with three substitutions AVX-512 needs: VBLENDPD becomes
// VBLENDMPD under K1 (odd lanes, 0xAA) or K2 (even lanes, 0x55); the mirror's
// four bins are reversed by VSHUFF64X2 $0x1B instead of VPERM2F128 swapping
// two; and VADDSUBPD, which has no 512-bit form, becomes a sign flip of the
// even lanes of the subtrahend (Z13, the real-lane mask) and an add: x − y
// and x + (−y) are the same IEEE operation, so the bits are the AVX2
// kernel's and the Go loop's. Z15, Z13, Z12 and Z14 hold the AVX2 kernel's
// Y15, Y13, Y12 and Y14 broadcast to eight lanes; VPXORQ flips signs
// (VXORPD on ZMM needs AVX512DQ). Only Z0..Z15 are written (Round 28: a
// dirty Z16..Z31 slowed code that never ran AVX-512).
func clUntangleStep512(j int, inverse bool) []string {
	z := func(i int) string { return fmt.Sprintf("Z%d", 4*j+i) }
	a, b, c, d := z(0), z(1), z(2), z(3)
	o := 64 * j
	in := []string{
		fmt.Sprintf("VMOVUPD %d(AX), %s", o, a),
		fmt.Sprintf("VMOVUPD %d(BX), %s", -o, b),
		fmt.Sprintf("VSHUFF64X2 $0x1b, %s, %s, %s", b, b, b),
		fmt.Sprintf("VADDPD %s, %s, %s", b, a, c), // s
		fmt.Sprintf("VSUBPD %s, %s, %s", b, a, d), // d
		fmt.Sprintf("VBLENDMPD %s, %s, K1, %s", d, c, a),
		fmt.Sprintf("VMULPD Z14, %s, %s", a, a), // xe
		fmt.Sprintf("VBLENDMPD %s, %s, K2, %s", d, c, b),
	}
	if !inverse {
		return append(in,
			fmt.Sprintf("VPXORQ Z13, %s, %s", b, b),
			fmt.Sprintf("VMULPD Z14, %s, %s", b, b),    // [xo.im, xo.re]
			fmt.Sprintf("VMOVDDUP %d(CX), %s", o, c),   // [wr, wr]
			fmt.Sprintf("VMOVDDUP %d(CX), %s", o+8, d), // [wi, wi]
			fmt.Sprintf("VMULPD %s, %s, %s", b, d, d),
			fmt.Sprintf("VPERMILPD $0x55, %s, %s", b, b), // xo
			fmt.Sprintf("VMULPD %s, %s, %s", b, c, c),
			fmt.Sprintf("VPXORQ Z13, %s, %s", d, d),
			fmt.Sprintf("VADDPD %s, %s, %s", d, c, c), // t
			fmt.Sprintf("VADDPD %s, %s, %s", c, a, b),
			fmt.Sprintf("VSUBPD %s, %s, %s", c, a, d),
			fmt.Sprintf("VPXORQ Z15, %s, %s", d, d),
			fmt.Sprintf("VSHUFF64X2 $0x1b, %s, %s, %s", d, d, d),
			fmt.Sprintf("VMOVUPD %s, %d(R9)", b, o),
			fmt.Sprintf("VMOVUPD %s, %d(R10)", d, -o),
		)
	}
	return append(in,
		fmt.Sprintf("VMULPD Z14, %s, %s", b, b),    // dd
		fmt.Sprintf("VMOVDDUP %d(CX), %s", o, d),   // [wr, wr]
		fmt.Sprintf("VMOVDDUP %d(CX), %s", o+8, c), // [wi, wi]
		fmt.Sprintf("VPXORQ Z12, %s, %s", c, c),    // [-wi, -wi]
		fmt.Sprintf("VMULPD %s, %s, %s", b, d, d),
		fmt.Sprintf("VPERMILPD $0x55, %s, %s", b, b),
		fmt.Sprintf("VMULPD %s, %s, %s", b, c, c),
		fmt.Sprintf("VPXORQ Z13, %s, %s", c, c),
		fmt.Sprintf("VADDPD %s, %s, %s", c, d, d), // xo
		fmt.Sprintf("VPERMILPD $0x55, %s, %s", d, b),
		fmt.Sprintf("VPXORQ Z15, %s, %s", b, b), // [xoi, -xor]
		fmt.Sprintf("VSUBPD %s, %s, %s", b, a, c),
		fmt.Sprintf("VADDPD %s, %s, %s", b, a, d),
		fmt.Sprintf("VPXORQ Z15, %s, %s", d, d),
		fmt.Sprintf("VSHUFF64X2 $0x1b, %s, %s, %s", d, d, d),
		fmt.Sprintf("VMOVUPD %s, %d(R9)", c, o),
		fmt.Sprintf("VMOVUPD %s, %d(R10)", d, -o),
	)
}

// genClUntangle512 emits clUntangleAVX512 or, when inverse,
// clRetangleAVX512: genSmallUntangle's kernels at 512 bits, smallUntangleSteps
// steps of four bins per iteration while that many quads are left, then one
// at a time, nothing at all for quads = 0. The last step reads tw[4·quads+1].re (the VMOVDDUP at tw+8),
// which the Go wrappers bound-check.
func genClUntangle512(f *emit.File, inverse bool) {
	name := "clUntangleAVX512"
	names := []string{"dst", "z", "tw", "k", "m", "quads"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}
	if inverse {
		name = "clRetangleAVX512"
		names = []string{"z", "x", "tw", "k", "h", "m", "quads"}
		types = []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64}
	}
	b := amd64.NewFunc(name, amd64.Layout(names, types, nil, nil), 0)
	b.LoadArg(names[0], "DI").LoadArg(names[1], "SI").LoadArg("tw", "CX").LoadArg("k", "R14").
		LoadArg("m", "DX").LoadArg("quads", "R8")
	// No step: return at once, before the first AVX-512 instruction and
	// without a VZEROUPPER, so the wrappers may call this on any CPU with
	// quads = 0 for the price of a call.
	b.Raw("TESTQ R8, R8").Raw("JZ none")
	b.Raw("SHLQ $4, DX").
		Raw("LEAQ 16(SI), AX").         // &in[1]
		Raw("LEAQ -64(SI)(DX*1), BX").  // &in[m-4]
		Raw("LEAQ 16(DI), R9").         // &out[1]
		Raw("LEAQ -64(DI)(DX*1), R10"). // &out[m-4]
		Raw("ADDQ $16, CX").            // &tw[1]
		Raw("MOVQ $0xaa, R11").Raw("KMOVB R11, K1").
		Raw("MOVQ $0x55, R11").Raw("KMOVB R11, K2").
		Raw("VBROADCASTF64X4 0(R14), Z15"). // imaginary-lane sign mask [0, -0]
		Raw("VPERMILPD $0x55, Z15, Z13").   // real-lane sign mask [-0, 0]
		Raw("VPORQ Z15, Z13, Z12")          // both lanes
	if inverse {
		b.LoadArg("h", "R15").Raw("VBROADCASTF64X4 (R15), Z14")
	} else {
		b.Raw("VBROADCASTF64X4 32(R14), Z14") // 0.5
	}
	adv := func(n int) {
		b.Raw("ADDQ $%d, AX", 64*n).
			Raw("SUBQ $%d, BX", 64*n).
			Raw("ADDQ $%d, R9", 64*n).
			Raw("SUBQ $%d, R10", 64*n).
			Raw("ADDQ $%d, CX", 64*n)
	}
	b.Raw("loop:").
		Raw("CMPQ R8, $%d", smallUntangleSteps).
		Raw("JLT tail")
	seqs := make([][]string, smallUntangleSteps)
	for j := range seqs {
		seqs[j] = clUntangleStep512(j, inverse)
	}
	for i := range seqs[0] {
		for j := range seqs {
			b.Raw("%s", seqs[j][i])
		}
	}
	adv(smallUntangleSteps)
	b.Raw("SUBQ $%d, R8", smallUntangleSteps).
		Raw("JMP loop").
		Raw("tail:").
		Raw("TESTQ R8, R8").
		Raw("JZ done")
	for _, ins := range clUntangleStep512(0, inverse) {
		b.Raw("%s", ins)
	}
	adv(1)
	b.Raw("DECQ R8").
		Raw("JMP tail").
		Raw("done:").
		Raw("VZEROUPPER").
		Ret()
	b.Raw("none:").Ret()
	f.Add(b.Func())
}

// ---------------------------------------------------------------------------
// Two-pass transforms on the kernel's own stack frame (Round 32), into
// zntwopass_amd64.s.
//
// A Stockham transform of two passes needs one n-point buffer between them.
// The fft package borrowed it from a sync.Pool, a round trip of about 16 ns,
// as long as a whole 64-point pass on Zen 3; a Go array on the stack would be
// zeroed on every call (1 KB at 64 points). znTwoPassAVX2(cc, ch, tw *complex128,
// k0, k1 *float64, ido, l1 int, pass, last uintptr) keeps the buffer in its
// own frame, uninitialised: it calls the first pass's kernel, at entry point
// pass, as (cc, buf, tw, k0, ido, 1), then the final pass's kernel, at entry
// point last, as (buf, ch, k1, l1). The buffer starts on a 64-byte boundary
// and holds up to znTwoPassMax points. Both kernels are the ones StockhamPass
// and StockhamPassLayout run, with the same arguments, so the result is the
// same bits.
//
// The entry points are the kernels' ABI0 addresses, which Go code cannot take
// (a func value of an assembly function is its ABIInternal wrapper):
// znKernelAddrs(t *[znAddrSlots]uintptr) stores them, the interleaved pass
// kernels of radix r at t[r], the final-pass kernels at t[21+r], the split
// kernels of radix r, mode m and direction d (0 forward) at
// t[42+(9d+r)·5+m].

// znTwoPassMax is the largest transform znTwoPassAVX2's frame holds, in
// complex128 points; it must match the kernels package's ZnTwoPassMax.
const znTwoPassMax = 256

// znRadices are the radices with AVX2 pass and final-pass kernels.
var znRadices = []int{2, 3, 4, 5, 8, 10, 12, 15, 16, 20}

// znSplitModes names the split kernels' layouts by mode (1..4).
var znSplitModes = []string{1: "IS", 2: "SS", 3: "SI", 4: "II"}

func genZnTwoPassFile() {
	f := emit.NewFile("amd64")
	genZnTwoPass(f)
	genZnThreePass(f)
	genZnKernelAddrs(f)
	// NO_LOCAL_POINTERS (the frame holds the buffer and the callees'
	// arguments, which are copies of znTwoPassAVX2's own) is in funcdata.h.
	s := strings.Replace(f.String(), "#include \"textflag.h\"\n", "#include \"textflag.h\"\n#include \"funcdata.h\"\n", 1)
	writeFile("zntwopass_amd64.s", s)
}

func genZnTwoPass(f *emit.File) {
	names := []string{"cc", "ch", "tw", "k0", "k1", "ido", "l1", "pass", "last"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Int64, amd64.Uint64, amd64.Uint64}
	// The frame: the callees' arguments at 0..47(SP), then the buffer from
	// the first 64-byte boundary at or above 48(SP).
	frame := (48 + 63 + 16*znTwoPassMax + 7) &^ 7
	// No NOSPLIT: the assembler inserts the stack check for the frame.
	b := amd64.NewFuncFlags("znTwoPassAVX2", amd64.Layout(names, types, nil, nil), frame, "")
	b.Raw("NO_LOCAL_POINTERS")
	buf := func(reg string) {
		b.Raw("LEAQ 111(SP), %s", reg).Raw("ANDQ $-64, %s", reg)
	}
	// The first pass: (cc, buf, tw, k0, ido, 1).
	b.LoadArg("cc", "AX").Raw("MOVQ AX, 0(SP)")
	buf("AX")
	b.Raw("MOVQ AX, 8(SP)")
	b.LoadArg("tw", "AX").Raw("MOVQ AX, 16(SP)")
	b.LoadArg("k0", "AX").Raw("MOVQ AX, 24(SP)")
	b.LoadArg("ido", "AX").Raw("MOVQ AX, 32(SP)")
	b.Raw("MOVQ $1, 40(SP)")
	b.LoadArg("pass", "AX").Raw("CALL AX")
	// The final pass: (buf, ch, k1, l1). The callee may have clobbered
	// every register.
	buf("AX")
	b.Raw("MOVQ AX, 0(SP)")
	b.LoadArg("ch", "AX").Raw("MOVQ AX, 8(SP)")
	b.LoadArg("k1", "AX").Raw("MOVQ AX, 16(SP)")
	b.LoadArg("l1", "AX").Raw("MOVQ AX, 24(SP)")
	b.LoadArg("last", "AX").Raw("CALL AX")
	b.Ret()
	f.Add(b.Func())
}

func genZnKernelAddrs(f *emit.File) {
	b := amd64.NewFunc("znKernelAddrs", amd64.Layout([]string{"t"}, []amd64.Type{amd64.Ptr}, nil, nil), 0)
	b.LoadArg("t", "DI")
	put := func(sym string, slot int) {
		b.Raw("LEAQ ·%s(SB), AX", sym).Raw("MOVQ AX, %d(DI)", 8*slot)
	}
	for _, r := range znRadices {
		put(fmt.Sprintf("skPass%dAVX2", r), r)
		put(fmt.Sprintf("skLast%dAVX2", r), 21+r)
	}
	for d, dir := range []string{"Fwd", "Inv"} {
		for _, r := range []int{4, 8} {
			for m := 1; m <= 4; m++ {
				put(fmt.Sprintf("skSplit%d%s%s", r, znSplitModes[m], dir), 42+(9*d+r)*5+m)
			}
		}
	}
	b.Ret()
	f.Add(b.Func())
}

// znThreePassMax is the largest transform znThreePassAVX2's frame holds, in
// complex128 points; it must match the kernels package's ZnThreePassMax.
const znThreePassMax = 512

// genZnThreePass writes znThreePassAVX2(cc, ch, x, tw0, tw1 *complex128, k0,
// k1, k2 *float64, ido0, ido1, l1b, l1c int, p0, p1, p2 uintptr), the
// three-pass counterpart of znTwoPassAVX2: pass 0 (entry point p0) reads cc
// and writes x, pass 1 (p1) reads x and writes a buffer B in the frame, the
// final pass (p2) reads B and writes ch, with the arguments StockhamPass
// would give each:
//
//	p0(cc, x, tw0, k0, ido0, 1)
//	p1(x, B, tw1, k1, ido1, l1b)
//	p2(B, ch, k2, l1c)
//
// x is ch when it does not alias cc (the fft package's out-of-place
// schedule for an odd pass count) or nil, which selects a second frame
// buffer B2 (in place, where pass 0 must not write over the input it is
// still reading). B and B2 start on 64-byte boundaries.
func genZnThreePass(f *emit.File) {
	names := []string{"cc", "ch", "x", "tw0", "tw1", "k0", "k1", "k2", "ido0", "ido1", "l1b", "l1c", "p0", "p1", "p2"}
	types := []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Ptr,
		amd64.Int64, amd64.Int64, amd64.Int64, amd64.Int64, amd64.Uint64, amd64.Uint64, amd64.Uint64}
	// The callees' arguments at 0..47(SP), x at 48(SP), then B from the
	// first 64-byte boundary at or above 56(SP), then B2.
	b2 := 16 * znThreePassMax
	frame := (56 + 63 + 2*b2 + 7) &^ 7
	b := amd64.NewFuncFlags("znThreePassAVX2", amd64.Layout(names, types, nil, nil), frame, "")
	b.Raw("NO_LOCAL_POINTERS")
	buf := func(reg string) {
		b.Raw("LEAQ 119(SP), %s", reg).Raw("ANDQ $-64, %s", reg)
	}
	// x, or B2 when x is nil, kept at 48(SP).
	b.LoadArg("x", "AX").
		Raw("TESTQ AX, AX").
		Raw("JNZ havex")
	buf("AX")
	b.Raw("ADDQ $%d, AX", b2).
		Label("havex")
	b.Raw("MOVQ AX, 48(SP)")
	// Pass 0: (cc, x, tw0, k0, ido0, 1).
	b.LoadArg("cc", "BX").Raw("MOVQ BX, 0(SP)").
		Raw("MOVQ AX, 8(SP)")
	b.LoadArg("tw0", "BX").Raw("MOVQ BX, 16(SP)")
	b.LoadArg("k0", "BX").Raw("MOVQ BX, 24(SP)")
	b.LoadArg("ido0", "BX").Raw("MOVQ BX, 32(SP)")
	b.Raw("MOVQ $1, 40(SP)")
	b.LoadArg("p0", "AX").Raw("CALL AX")
	// Pass 1: (x, B, tw1, k1, ido1, l1b).
	b.Raw("MOVQ 48(SP), AX").Raw("MOVQ AX, 0(SP)")
	buf("AX")
	b.Raw("MOVQ AX, 8(SP)")
	b.LoadArg("tw1", "AX").Raw("MOVQ AX, 16(SP)")
	b.LoadArg("k1", "AX").Raw("MOVQ AX, 24(SP)")
	b.LoadArg("ido1", "AX").Raw("MOVQ AX, 32(SP)")
	b.LoadArg("l1b", "AX").Raw("MOVQ AX, 40(SP)")
	b.LoadArg("p1", "AX").Raw("CALL AX")
	// The final pass: (B, ch, k2, l1c).
	buf("AX")
	b.Raw("MOVQ AX, 0(SP)")
	b.LoadArg("ch", "AX").Raw("MOVQ AX, 8(SP)")
	b.LoadArg("k2", "AX").Raw("MOVQ AX, 16(SP)")
	b.LoadArg("l1c", "AX").Raw("MOVQ AX, 24(SP)")
	b.LoadArg("p2", "AX").Raw("CALL AX")
	b.Ret()
	f.Add(b.Func())
}

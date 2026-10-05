//go:build ignore

// Command gen produces, via go-asmgen, cmul_arm64.s (the NEON pointwise
// complex multiply kernel) and stockham_arm64.s (the NEON Stockham pass
// kernels, genStockhamPass4 and genStockhamLast4 below). Run with: go run
// gen.go (or `go generate` from the
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
		Raw("VLD2 (R0), [V0.D2, V1.D2]").   // V0=[ar0,ar1] V1=[ai0,ai1]
		Raw("VLD2 (R1), [V2.D2, V3.D2]").   // V2=[br0,br1] V3=[bi0,bi1]
		Raw("VEOR V4.B16, V4.B16, V4.B16"). // re acc = 0
		Raw("VEOR V5.B16, V5.B16, V5.B16"). // im acc = 0
		Raw("VFMLA V2.D2, V0.D2, V4.D2").   // re += ar*br
		Raw("VFMLS V3.D2, V1.D2, V4.D2").   // re -= ai*bi  (fused, matches oracle FMSUBD)
		Raw("VFMLA V3.D2, V0.D2, V5.D2").   // im += ar*bi
		Raw("VFMLA V2.D2, V1.D2, V5.D2").   // im += ai*br  (fused, matches oracle FMADDD)
		Raw("VST2 [V4.D2, V5.D2], (R0)").
		Raw("ADD $32, R0").
		Raw("ADD $32, R1").
		Raw("SUB $2, R2").
		Raw("B loop2").
		Raw("tail:").
		Raw("CBZ R2, done").
		Raw("FMOVD (R0), F0").        // ar
		Raw("FMOVD 8(R0), F1").       // ai
		Raw("FMOVD (R1), F2").        // br
		Raw("FMOVD 8(R1), F3").       // bi
		Raw("FMULD F2, F0, F4").      // ar*br
		Raw("FMSUBD F1, F4, F3, F4"). // F4 - ai*bi  (fused, matches oracle)
		Raw("FMULD F3, F0, F5").      // ar*bi
		Raw("FMADDD F2, F5, F1, F5"). // F5 + ai*br  (fused, matches oracle)
		Raw("FMOVD F4, (R0)").
		Raw("FMOVD F5, 8(R0)").
		Raw("done:").
		Ret()
	f.Add(b.Func())

	write("cmul_arm64.s", f)

	sk := emit.NewFile("arm64")
	for _, inverse := range []bool{false, true} {
		genStockhamPass4(sk, inverse)
		genStockhamLast4(sk, inverse)
	}
	write("stockham_arm64.s", sk)
}

func write(name string, f *emit.File) {
	if err := os.WriteFile(name, []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

// The Stockham pass kernels run stockham.go's radix-4 pass (pass4) and its
// final pass (pass4last) two points per NEON register, deinterleaved: VLD2
// splits two complex128 into a register of real parts and one of imaginary
// parts, every lane then runs exactly the scalar operation the Go pass runs on
// one point, and VST2 interleaves on store.
//
// Bit identity with the Go pass, which gc compiles on arm64 with fused
// multiply-adds (read with go build -gcflags=-S): the additions and
// subtractions of the butterfly are plain FADDD/FSUBD, so VFADD/VFSUB; the ±i
// rotation rotS multiplies by s = ±1, which is exact, fused or not, so it is
// written as an exchange of the real and imaginary registers and the choice of
// add or subtract (x - y is x + (-y) in IEEE 754, signed zeros included); and
// the twiddle product y·w compiles, per point, to
//
//	re = FMULD(yr, wr), then FMSUBD: re - yi·wi, fused
//	im = FMULD(yi, wr), then FMADDD: im + yr·wi, fused
//
// that is, the rounded products are yr·wr and yi·wr and the fused ones yi·wi
// and yr·wi. The kernels do the same with VFMUL, VFMLS and VFMLA. (cmul.go's
// CMulScalar fuses the other imaginary product, ai·br: which product gc rounds
// first depends on the code around it, so each kernel copies the sequence of the
// pass it replaces, and the tests compare them bit for bit.)
//
// Twiddles are read from one sequential stream (kernels.StockhamTwiddles on
// arm64), already split: for each pair of points i, i+1, the four float64
// w1r(i) w1r(i+1) w1i(i) w1i(i+1), then the same for w2 and w3, so plain VLD1
// loads them without a deinterleave. The pair i = 0, 1 is multiplied with
// w(0) = 1 like the others; the scalar pass does not multiply point 0, and a
// multiply by one is not exact on signed zeros and infinities, so lane 0 of the
// untwiddled result is put back (VMOV element) before the store.
//
// Register use. R0 cc, R1 ch, R2 tw, R3 ido, R4 l1; R16 = ido·16 (one input
// stream), R17 = l1·ido·16 (one output stream); R19 the block's input, R20 its
// output; R6-R9 the input streams, R10-R13 the output streams, R14 the twiddle
// cursor, R5 and R15 the block and pair counters.

// pass4Pair emits one pair of points of the radix-4 pass. first marks the pair
// i = 0, 1, whose lane 0 is restored untwiddled.
func pass4Pair(b *arm64.Builder, inverse, first bool) {
	for j := 0; j < 4; j++ { // x_j: real parts in V(2j), imaginary in V(2j+1)
		b.Raw("VLD2.P 32(R%d), [V%d.D2, V%d.D2]", 6+j, 2*j, 2*j+1)
	}
	b.Raw("VLD1.P 64(R14), [V20.D2, V21.D2, V22.D2, V23.D2]") // w1 re, im, w2 re, im
	b.Raw("VLD1.P 32(R14), [V24.D2, V25.D2]")                 // w3 re, im
	bfly4(b)
	// Forward: y1 = (A, C), y3 = (B, D). Inverse, rotS(t4) = (-t4i, t4r):
	// y1 = (B, D), y3 = (A, C).
	y1, y3 := 0, 2
	if inverse {
		y1, y3 = 2, 0
	}
	cmul := func(d, y, w int) {
		b.VFMUL2D(d, y, w).VFMLS2D(d, y+1, w+1)     // re = yr·wr - yi·wi, yi·wi fused
		b.VFMUL2D(d+1, y+1, w).VFMLA2D(d+1, y, w+1) // im = yi·wr + yr·wi, yr·wi fused
	}
	cmul(26, y1, 20)
	cmul(28, 18, 22)
	cmul(30, y3, 24)
	if first {
		for _, m := range [][2]int{{y1, 26}, {y1 + 1, 27}, {18, 28}, {19, 29}, {y3, 30}, {y3 + 1, 31}} {
			b.Raw("VMOV V%d.D[0], V%d.D[0]", m[0], m[1])
		}
	}
	b.Raw("VST2.P [V16.D2, V17.D2], 32(R10)").
		Raw("VST2.P [V26.D2, V27.D2], 32(R11)").
		Raw("VST2.P [V28.D2, V29.D2], 32(R12)").
		Raw("VST2.P [V30.D2, V31.D2], 32(R13)")
}

// genStockhamPass4 emits skPass4NEON / skPass4NEONInv(cc, ch, tw *complex128,
// ido, l1 int): the radix-4 pass for an even ido >= 2.
func genStockhamPass4(f *emit.File, inverse bool) {
	name := "skPass4NEON"
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "tw", "ido", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Int64}, nil, nil,
	)
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R0").LoadArg("ch", "R1").LoadArg("tw", "R2").
		LoadArg("ido", "R3").LoadArg("l1", "R4").
		Raw("LSL $4, R3, R16").
		Raw("MUL R4, R16, R17").
		Raw("MOVD R0, R19").
		Raw("MOVD R1, R20").
		Raw("MOVD R4, R5")
	b.Label("kloop").
		Raw("MOVD R19, R6").
		Raw("ADD R16, R6, R7").
		Raw("ADD R16, R7, R8").
		Raw("ADD R16, R8, R9").
		Raw("MOVD R20, R10").
		Raw("ADD R17, R10, R11").
		Raw("ADD R17, R11, R12").
		Raw("ADD R17, R12, R13").
		Raw("MOVD R2, R14")
	pass4Pair(b, inverse, true)
	b.Raw("LSR $1, R3, R15").
		Raw("SUBS $1, R15, R15").
		Raw("BEQ knext")
	b.Label("iloop")
	pass4Pair(b, inverse, false)
	b.Raw("SUBS $1, R15, R15").
		Raw("BNE iloop")
	b.Label("knext").
		Raw("ADD R16<<2, R19, R19").
		Raw("ADD R16, R20, R20").
		Raw("SUBS $1, R5, R5").
		Raw("BNE kloop").
		Ret()
	f.Add(b.Func())
}

// genStockhamLast4 emits skLast4NEON / skLast4NEONInv(cc, ch *complex128, l1
// int): the final radix-4 pass (ido == 1) for an even l1, two blocks per
// iteration. Block k is the four points cc[4k..4k+3]; two blocks are loaded
// as they lie and their points paired across blocks with VZIP1/VZIP2 (real
// parts of point j of blocks k and k+1, then imaginary parts), and output j
// of both blocks, ch[j·l1+k], ch[j·l1+k+1], is one VST2. No twiddles.
func genStockhamLast4(f *emit.File, inverse bool) {
	name := "skLast4NEON"
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64}, nil, nil,
	)
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R0").LoadArg("ch", "R1").LoadArg("l1", "R4").
		Raw("LSL $4, R4, R17").
		Raw("MOVD R1, R10").
		Raw("ADD R17, R10, R11").
		Raw("ADD R17, R11, R12").
		Raw("ADD R17, R12, R13").
		Raw("LSR $1, R4, R5")
	b.Label("loop").
		Raw("VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]"). // block k: x0..x3
		Raw("VLD1.P 64(R0), [V24.D2, V25.D2, V26.D2, V27.D2]")  // block k+1
	for j := 0; j < 4; j++ {
		b.Raw("VZIP1 V%d.D2, V%d.D2, V%d.D2", 24+j, 20+j, 2*j).
			Raw("VZIP2 V%d.D2, V%d.D2, V%d.D2", 24+j, 20+j, 2*j+1)
	}
	bfly4(b)
	y1, y3 := 0, 2
	if inverse {
		y1, y3 = 2, 0
	}
	b.Raw("VST2.P [V16.D2, V17.D2], 32(R10)").
		Raw("VST2.P [V%d.D2, V%d.D2], 32(R11)", y1, y1+1).
		Raw("VST2.P [V18.D2, V19.D2], 32(R12)").
		Raw("VST2.P [V%d.D2, V%d.D2], 32(R13)", y3, y3+1).
		Raw("SUBS $1, R5, R5").
		Raw("BNE loop").
		Ret()
	f.Add(b.Func())
}

// bfly4 emits stockham.go's bfly4 on x0..x3 in V0-V7 (real, imaginary), up to
// the ±i rotation: y0 in V16/V17, y2 in V18/V19, and A, C, B, D in V0-V3,
// where forward y1 = (A, C), y3 = (B, D) and inverse the other way round.
func bfly4(b *arm64.Builder) {
	b.VFADD2D(8, 0, 4).VFADD2D(9, 1, 5)     // t2 = x0 + x2
	b.VFSUB2D(10, 0, 4).VFSUB2D(11, 1, 5)   // t1 = x0 - x2
	b.VFADD2D(12, 2, 6).VFADD2D(13, 3, 7)   // t3 = x1 + x3
	b.VFSUB2D(14, 2, 6).VFSUB2D(15, 3, 7)   // t4 = x1 - x3
	b.VFADD2D(16, 8, 12).VFADD2D(17, 9, 13) // y0 = t2 + t3
	b.VFSUB2D(18, 8, 12).VFSUB2D(19, 9, 13) // y2 = t2 - t3
	// t1 ± rotS(t4); forward rotS(t4) = (t4i, -t4r), inverse (-t4i, t4r).
	b.VFADD2D(0, 10, 15) // A = t1r + t4i
	b.VFSUB2D(1, 11, 14) // C = t1i - t4r
	b.VFSUB2D(2, 10, 15) // B = t1r - t4i
	b.VFADD2D(3, 11, 14) // D = t1i + t4r
}

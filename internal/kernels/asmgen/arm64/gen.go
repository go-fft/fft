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
		genStockhamPass(sk, 4, inverse, pass4Pair)
		genStockhamPass(sk, 8, inverse, pass8Pair)
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
func pass4Pair(b *arm64.Builder, in, out []string, inverse bool, st step) {
	for j := 0; j < 4; j++ { // x_j: real parts in V(2j), imaginary in V(2j+1)
		st.load(b, in[j], 2*j)
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
	if st == firstPair {
		for _, m := range [][2]int{{y1, 26}, {y1 + 1, 27}, {18, 28}, {19, 29}, {y3, 30}, {y3 + 1, 31}} {
			b.Raw("VMOV V%d.D[0], V%d.D[0]", m[0], m[1])
		}
	}
	for j, v := range []int{16, 26, 28, 30} {
		st.store(b, v, out[j])
	}
}

// step is what one call of a pass body emits: the pair of points i = 0, 1
// (whose lane 0 is restored untwiddled), a later pair, or the last point of an
// odd ido. A lone point is loaded into lane 0 with a scalar FLDPD (which
// zeroes lane 1), runs through the same vector code with a dummy lane 1, and
// lane 0 is stored with FSTPD; its twiddles come as a pair whose second half is
// zero.
type step int

const (
	firstPair step = iota
	nextPair
	lastPoint
)

// load loads the next point(s) of the stream at ptr: real parts into V(v),
// imaginary parts into V(v+1).
func (st step) load(b *arm64.Builder, ptr string, v int) {
	if st == lastPoint {
		b.Raw("FLDPD (%s), (F%d, F%d)", ptr, v, v+1)
		return
	}
	b.Raw("VLD2.P 32(%s), [V%d.D2, V%d.D2]", ptr, v, v+1)
}

// store stores V(v) (real parts) and V(v+1) (imaginary parts) to the stream
// at ptr.
func (st step) store(b *arm64.Builder, v int, ptr string) {
	if st == lastPoint {
		b.Raw("FSTPD (F%d, F%d), (%s)", v, v+1, ptr)
		return
	}
	b.Raw("VST2.P [V%d.D2, V%d.D2], 32(%s)", v, v+1, ptr)
}

// skStreams holds, per radix, the registers that walk the pass's input and
// output streams (the rest of the register use is in the comment above).
var skStreams = map[int][2][]string{
	4: {{"R6", "R7", "R8", "R9"}, {"R10", "R11", "R12", "R13"}},
	8: {{"R0", "R1", "R4", "R6", "R7", "R8", "R9", "R10"}, {"R11", "R12", "R13", "R21", "R22", "R23", "R24", "R25"}},
}

// genStockhamPass emits skPass<r>NEON / skPass<r>NEONInv(cc, ch, tw
// *complex128, ido, l1 int): the radix-r pass for an even ido >= 2. pair emits
// one pair of points; first marks the pair i = 0, 1.
func genStockhamPass(f *emit.File, r int, inverse bool, pair func(b *arm64.Builder, in, out []string, inverse bool, st step)) {
	name := fmt.Sprintf("skPass%dNEON", r)
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "tw", "ido", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Int64}, nil, nil,
	)
	in, out := skStreams[r][0], skStreams[r][1]
	shift := map[int]int{4: 2, 8: 3}[r]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R19").LoadArg("ch", "R20").LoadArg("tw", "R2").
		LoadArg("ido", "R3").LoadArg("l1", "R5").
		Raw("LSL $4, R3, R16"). // S: one input stream
		Raw("MUL R5, R16, R17") // OS: one output stream
	if r == 8 {
		b.Raw("FMOVD $(0.707106781186547524400844362104849), F18"). // √2/2
										Raw("VDUP V18.D[0], V18.D2")
	}
	b.Label("kloop").Raw("MOVD R19, %s", in[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R16, %s, %s", in[j-1], in[j])
	}
	b.Raw("MOVD R20, %s", out[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("MOVD R2, R14")
	pair(b, in, out, inverse, firstPair)
	b.Raw("LSR $1, R3, R15").
		Raw("SUBS $1, R15, R15").
		Raw("BEQ knext")
	b.Label("iloop")
	pair(b, in, out, inverse, nextPair)
	b.Raw("SUBS $1, R15, R15").
		Raw("BNE iloop")
	b.Label("knext").
		Raw("TBZ $0, R3, knext2") // odd ido: one point left
	pair(b, in, out, inverse, lastPoint)
	b.Label("knext2").
		Raw("ADD R16<<%d, R19, R19", shift).
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

// pass8Pair emits one pair of points of the radix-8 pass: stockham.go's pass8
// loop body, which gc compiles (-gcflags=-S) with these fusions beyond the
// twiddle product's: every ±s·z (the ±i rotations, s = ±1) is fused into the
// add or subtract that uses it, which is exact; the ∓45°/∓135° rotations are
// computed unscaled, P = a5r - s·a5i, Q = a5i + s·a5r, R = -a7r - s·a7i
// (FNMADDD), T = -a7i + s·a7r (FNMSUBD), and their scaling by √2/2 is fused
// into the final add or subtract: y1 = a4 + h·(P, Q), y5 = a4 - h·(P, Q), y3 =
// a6 + h·(R, T), y7 = a6 - h·(R, T), each one FMADDD/FMSUBD. The kernel does
// the same with VFMLA/VFMLS on a copy of a4 and a6. R and T are sums of two
// negated operands, so they are written (-a) - b with a VFNEG, never -(a + b),
// which differs on a zero sum. h, √2/2, is in V18 for the whole call.
func pass8Pair(b *arm64.Builder, in, out []string, inverse bool, st step) {
	fwd := !inverse
	add := func(d, n, m int) { b.VFADD2D(d, n, m) }
	sub := func(d, n, m int) { b.VFSUB2D(d, n, m) }
	pm := func(d, n, m int, plus bool) {
		if plus {
			add(d, n, m)
		} else {
			sub(d, n, m)
		}
	}
	aMinusSz := func(d, a, z int) { pm(d, a, z, fwd) } // a - s·z: forward a + z, inverse a - z
	aPlusSz := func(d, a, z int) { pm(d, a, z, !fwd) } // a + s·z
	for j := 0; j < 8; j++ {                           // x_j in V(2j), V(2j+1)
		st.load(b, in[j], 2*j)
	}
	add(16, 2, 10) // a1 = x1 + x5
	add(17, 3, 11)
	sub(2, 2, 10) // a5 = x1 - x5
	sub(3, 3, 11)
	add(28, 6, 14) // a3 = x3 + x7
	add(29, 7, 15)
	sub(6, 6, 14) // a7 = x3 - x7
	sub(7, 7, 15)
	add(20, 16, 28) // a1 = a1 + a3
	add(21, 17, 29)
	sub(16, 16, 28) // a3 = a1 - a3, then rotS
	sub(17, 17, 29)
	aMinusSz(22, 2, 7) // a5 = a5 + rotS(a7)
	aPlusSz(23, 3, 6)
	aPlusSz(24, 2, 7) // a7 = a5 - rotS(a7)
	aMinusSz(25, 3, 6)
	aMinusSz(2, 22, 23) // P
	aPlusSz(3, 23, 22)  // Q
	if fwd {
		sub(6, 25, 24) // R = -a7r + a7i
		b.VFNEG2D(7, 25)
		sub(7, 7, 24) // T = -a7i - a7r
	} else {
		b.VFNEG2D(6, 24)
		sub(6, 6, 25)  // R = -a7r - a7i
		sub(7, 24, 25) // T = -a7i + a7r
	}
	add(22, 0, 8) // a0 = x0 + x4
	add(23, 1, 9)
	sub(0, 0, 8) // a4 = x0 - x4
	sub(1, 1, 9)
	add(24, 4, 12) // a2 = x2 + x6
	add(25, 5, 13)
	sub(4, 4, 12) // a6 = x2 - x6
	sub(5, 5, 13)
	add(26, 22, 24) // a0 = a0 + a2
	add(27, 23, 25)
	sub(22, 22, 24) // a2 = a0 - a2
	sub(23, 23, 25)
	aMinusSz(8, 0, 5) // a4 = a4 + rotS(a6)
	aPlusSz(9, 1, 4)
	aPlusSz(10, 0, 5) // a6 = a4 - rotS(a6)
	aMinusSz(11, 1, 4)
	add(12, 26, 20) // y0 = a0 + a1
	add(13, 27, 21)
	st.store(b, 12, out[0])
	// Outputs 1..7, in twiddle-table order: y into (yr, yi), times w_j.
	for j := 1; j < 8; j++ {
		yr, yi := 14, 15
		switch j {
		case 1: // a4 + h·(P, Q)
			b.Raw("VMOV V8.B16, V14.B16").Raw("VMOV V9.B16, V15.B16")
			b.VFMLA2D(14, 2, 18).VFMLA2D(15, 3, 18)
		case 2: // a2 + rotS(a3)
			aMinusSz(14, 22, 17)
			aPlusSz(15, 23, 16)
		case 3: // a6 + h·(R, T)
			b.Raw("VMOV V10.B16, V14.B16").Raw("VMOV V11.B16, V15.B16")
			b.VFMLA2D(14, 6, 18).VFMLA2D(15, 7, 18)
		case 4: // a0 - a1
			sub(14, 26, 20)
			sub(15, 27, 21)
		case 5: // a4 - h·(P, Q), in place
			b.VFMLS2D(8, 2, 18).VFMLS2D(9, 3, 18)
			yr, yi = 8, 9
		case 6: // a2 - rotS(a3)
			aPlusSz(14, 22, 17)
			aMinusSz(15, 23, 16)
		case 7: // a6 - h·(R, T), in place
			b.VFMLS2D(10, 6, 18).VFMLS2D(11, 7, 18)
			yr, yi = 10, 11
		}
		b.Raw("VLD1.P 32(R14), [V28.D2, V29.D2]")
		b.VFMUL2D(30, yr, 28).VFMLS2D(30, yi, 29)
		b.VFMUL2D(31, yi, 28).VFMLA2D(31, yr, 29)
		if st == firstPair {
			b.Raw("VMOV V%d.D[0], V30.D[0]", yr).Raw("VMOV V%d.D[0], V31.D[0]", yi)
		}
		st.store(b, 30, out[j])
	}
}

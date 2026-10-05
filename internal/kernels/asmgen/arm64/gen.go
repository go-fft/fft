//go:build ignore

// Command gen produces, via go-asmgen, cmul_arm64.s (the NEON pointwise
// complex multiply kernel) and stockham_arm64.s (the NEON Stockham pass
// kernels, genStockhamPass and genStockhamLast below). Run with: go run gen.go
// (or `go generate` from the kernels package).
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
//	re = VFMUL ar,br (rounded); VFMLS ai,bi (re -= ai*bi, fused)
//	im = VFMUL ar,bi (rounded); VFMLA ai,br (im += ai*br, fused)
//
// A naive non-fused NEON (separate multiply then add) — or, equally, a fused
// kernel whose fusion form differs from the oracle's — diverges by up to 1 ULP;
// the random SIMD-vs-scalar test catches it, so this is validated, not assumed.
//
// Until Go 1.27 the arm64 assembler had no VFMUL, and the kernel formed each
// rounded product as VFMLA into a zeroed accumulator. That is the product
// rounded once, but not on signed zeros: +0 + (-0) is +0, so (-0)·1 came out
// +0 where the oracle's FMULD gives -0 (found 2026-10-05,
// TestCMulSignedZeroNEON). VFMUL is exact there too.
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
		VFMUL2D(4, 0, 2).                 // re = ar*br
		VFMLS2D(4, 1, 3).                 // re -= ai*bi  (fused, matches oracle FMSUBD)
		VFMUL2D(5, 0, 3).                 // im = ar*bi
		VFMLA2D(5, 1, 2).                 // im += ai*br  (fused, matches oracle FMADDD)
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
		for _, r := range []int{2, 3, 4, 5, 8} {
			genStockhamPass(sk, r, inverse)
		}
		for _, r := range []int{2, 3, 4, 5} {
			genStockhamLast(sk, r, inverse)
		}
	}
	for _, inverse := range []bool{false, true} {
		for _, r := range []int{2, 3, 4, 5, 8} {
			genStockhamBatchNEON(sk, r, inverse)
		}
	}
	for _, inverse := range []bool{false, true} {
		for _, lay := range []splitLayout{{out: true}, {in: true, out: true}} {
			for _, r := range []int{2, 4, 8} {
				genStockhamPassSplit(sk, r, inverse, lay)
			}
		}
		for _, r := range []int{2, 4} {
			genStockhamLastSplit(sk, r, inverse)
		}
	}
	write("stockham_arm64.s", sk)

	genF32StockhamFile()
}

func write(name string, f *emit.File) {
	if err := os.WriteFile(name, []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

// The Stockham kernels run stockham.go's radix-2, 3, 4, 5 and 8 passes
// (pass2 ... pass8) and the radix-2, 3, 4 and 5 final passes (pass2last ...
// pass5last) two points per NEON register, deinterleaved: VLD2
// splits two complex128 into a register of real parts and one of imaginary
// parts, every lane then runs exactly the scalar operation the Go pass runs on
// one point, and VST2 interleaves on store.
//
// Bit identity with the Go passes, which gc compiles on arm64 with fused
// multiply-adds. What gc fuses was read from go build -gcflags=-S (Go 1.27.1);
// each kernel copies it, and the tests compare kernel and Go pass bit for bit,
// signed zeros and infinities included:
//
//   - The butterfly's additions and subtractions are plain FADDD/FSUBD, so
//     VFADD/VFSUB.
//   - The ±i rotation rotS multiplies by s = ±1 and gc fuses that product into
//     the add or subtract that uses it. A product by ±1 is exact, so the result
//     is that of a plain add or subtract of ±z: the kernels exchange the real
//     and imaginary registers and choose VFADD or VFSUB (x - y is x + (-y) in
//     IEEE 754, signed zeros included). A sum of two negated operands, which
//     gc forms with FNMADDD/FNMSUBD, is written (-a) - b with a VFNEG, never
//     -(a + b), which differs on a zero sum.
//   - The twiddle product y·w compiles to re = FMULD(yr, wr) then FMSUBD
//     re - yi·wi, and, in pass3, pass4, pass5 and pass8, im = FMULD(yi, wr)
//     then FMADDD im + yr·wi: the products yr·wr and yi·wr are rounded, yi·wi
//     and yr·wi fused. In pass2 the imaginary part is the other way round, im =
//     FMULD(yr, wi) then FMADDD im + yi·wr (as in cmul.go's CMulScalar): which
//     product gc rounds depends on the code around it, so each kernel copies
//     the pass it replaces, with VFMUL, VFMLS and VFMLA.
//   - Radix 3 fuses 0.5·t1 into x0 - 0.5·t1 and rounds sin(2π/3)·t2.
//   - Radix 5: c·x + c'·y rounds c'·y and fuses c·x; c·x - c'·y rounds c·x and
//     fuses c'·y.
//   - Radix 8 computes its ∓45°/∓135° rotations unscaled and fuses their
//     √2/2 into the final add or subtract (see bfly8).
//
// Twiddles are read from one sequential stream (kernels.StockhamTwiddles on
// arm64), already split: for each pair of points i, i+1 and each j, the four
// float64 re(w_j(i)) re(w_j(i+1)) im(w_j(i)) im(w_j(i+1)), so a VLD1 loads
// them as they are used. The pair i = 0, 1 is multiplied with w(0) = 1 like the
// others; the Go pass does not multiply point 0, and a multiply by one is not
// exact on signed zeros and infinities, so lane 0 of the untwiddled result is
// put back (VMOV element) before the store. An odd ido ends with one point
// alone (see step).
//
// Register use in a pass kernel: R2 tw, R3 ido, R5 the block counter, R14 the
// twiddle cursor, R15 the pair counter, R16 = ido·16 (one input stream), R17 =
// l1·ido·16 (one output stream), R4 = r·ido·16 (one input block, radix 3 and
// 5), R19 the block's input, R20 its output, and skStreams the streams. V28
// and V29 hold a twiddle, V30 and V31 the product; the constants are in V18
// (radix 8) and V24-V27 (radix 3 and 5).

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

// skStreams holds, per radix, the registers that walk a pass's input and
// output streams.
var skStreams = map[int][2][]string{
	2: {{"R6", "R7"}, {"R10", "R11"}},
	3: {{"R6", "R7", "R8"}, {"R10", "R11", "R12"}},
	4: {{"R6", "R7", "R8", "R9"}, {"R10", "R11", "R12", "R13"}},
	5: {{"R6", "R7", "R8", "R9", "R10"}, {"R11", "R12", "R13", "R21", "R22"}},
	8: {{"R0", "R1", "R4", "R6", "R7", "R8", "R9", "R10"}, {"R11", "R12", "R13", "R21", "R22", "R23", "R24", "R25"}},
}

// skConsts emits the constants a radix's butterfly reads, each duplicated
// into both lanes of its register.
func skConsts(b *arm64.Builder, r int) {
	dup := func(v int, c string) {
		b.Raw("FMOVD $(%s), F%d", c, v).Raw("VDUP V%d.D[0], V%d.D2", v, v)
	}
	switch r {
	case 3:
		dup(24, "0.5")
		dup(25, "0.8660254037844386467637231707529361834714026269051903140279") // sin(2π/3)
	case 5:
		dup(24, "0.30901699437494742410229341718281905886015458990288")  // c51 = cos(2π/5)
		dup(25, "-0.80901699437494742410229341718281905886015458990289") // c52 = cos(4π/5)
		dup(26, "0.95105651629515357211643933337938214340569863412575")  // s51 = sin(2π/5)
		dup(27, "0.58778525229247312916870595463907276859765243764314")  // s52 = sin(4π/5)
	case 8:
		dup(18, "0.707106781186547524400844362104849") // √2/2
	}
}

// bfly emits the radix-r butterfly on x_j in V(2j), V(2j+1), calling out(j,
// yr, yi) for every output j in ascending order once y_j is in (yr, yi).
// Between calls, out may use V28-V31.
func bfly(b *arm64.Builder, r int, inverse bool, out func(j, yr, yi int)) {
	map[int]func(*arm64.Builder, bool, func(j, yr, yi int)){2: bfly2, 3: bfly3, 4: bfly4, 5: bfly5, 8: bfly8}[r](b, inverse, out)
}

// sops returns the add, the subtract, and the two ±s·z forms: aMinusSz(d, a,
// z) is a - s·z (forward a + z, inverse a - z) and aPlusSz a + s·z, s being
// the direction sign of stockham.go (-1 forward).
func sops(b *arm64.Builder, inverse bool) (add, sub, aMinusSz, aPlusSz func(d, n, m int)) {
	add = func(d, n, m int) { b.VFADD2D(d, n, m) }
	sub = func(d, n, m int) { b.VFSUB2D(d, n, m) }
	pm := func(d, n, m int, plus bool) {
		if plus {
			add(d, n, m)
		} else {
			sub(d, n, m)
		}
	}
	aMinusSz = func(d, a, z int) { pm(d, a, z, !inverse) }
	aPlusSz = func(d, a, z int) { pm(d, a, z, inverse) }
	return
}

// bfly2 is the radix-2 butterfly of pass2 and pass2last: y0 = x0 + x1, y1 =
// x0 - x1.
func bfly2(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	b.VFADD2D(4, 0, 2).VFADD2D(5, 1, 3)
	out(0, 4, 5)
	b.VFSUB2D(4, 0, 2).VFSUB2D(5, 1, 3)
	out(1, 4, 5)
}

// bfly3 is stockham.go's bfly3: t1 = x1 + x2, t2 = x1 - x2, ca = x0 - 0.5·t1
// (fused), q = sin(2π/3)·t2 (rounded), y0 = x0 + t1, y1 = ca + rotS(q), y2 =
// ca - rotS(q).
func bfly3(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := sops(b, inverse)
	add(6, 2, 4) // t1
	add(7, 3, 5)
	sub(2, 2, 4) // t2
	sub(3, 3, 5)
	add(8, 0, 6) // y0
	add(9, 1, 7)
	out(0, 8, 9)
	b.VFMLS2D(0, 6, 24).VFMLS2D(1, 7, 24) // ca, in place on x0
	b.VFMUL2D(2, 2, 25).VFMUL2D(3, 3, 25) // q, in place on t2
	aMinusSz(10, 0, 3)                    // y1 = ca + rotS(q)
	aPlusSz(11, 1, 2)
	out(1, 10, 11)
	aPlusSz(10, 0, 3) // y2 = ca - rotS(q)
	aMinusSz(11, 1, 2)
	out(2, 10, 11)
}

// bfly4 is stockham.go's bfly4: t2 = x0 + x2, t1 = x0 - x2, t3 = x1 + x3, t4
// = x1 - x3, y0 = t2 + t3, y1 = t1 + rotS(t4), y2 = t2 - t3, y3 = t1 - rotS(t4).
func bfly4(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := sops(b, inverse)
	add(8, 0, 4) // t2
	add(9, 1, 5)
	sub(10, 0, 4) // t1
	sub(11, 1, 5)
	add(12, 2, 6) // t3
	add(13, 3, 7)
	sub(14, 2, 6) // t4
	sub(15, 3, 7)
	add(16, 8, 12) // y0
	add(17, 9, 13)
	out(0, 16, 17)
	aMinusSz(0, 10, 15) // y1 = t1 + rotS(t4)
	aPlusSz(1, 11, 14)
	out(1, 0, 1)
	sub(16, 8, 12) // y2
	sub(17, 9, 13)
	out(2, 16, 17)
	aPlusSz(0, 10, 15) // y3 = t1 - rotS(t4)
	aMinusSz(1, 11, 14)
	out(3, 0, 1)
}

// bfly5 is stockham.go's bfly5 (inlined by hand in pass5): t1 = x1 + x4, t2 =
// x1 - x4, t3 = x2 + x3, t4 = x2 - x3; r1 = a + (c51·t1 + c52·t3), r2 = a +
// (c52·t1 + c51·t3), u = s51·t2 + s52·t4, v = s52·t2 - s51·t4, each a product
// rounded and the other fused as described above; y0 = (a + t1) + t3, y1 = r1
// + rotS(u), y2 = r2 + rotS(v), y3 = r2 - rotS(v), y4 = r1 - rotS(u).
func bfly5(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := sops(b, inverse)
	add(10, 2, 8) // t1
	add(11, 3, 9)
	sub(2, 2, 8) // t2
	sub(3, 3, 9)
	add(12, 4, 6) // t3
	add(13, 5, 7)
	sub(4, 4, 6) // t4
	sub(5, 5, 7)
	add(14, 0, 10) // y0 = (a + t1) + t3
	add(15, 1, 11)
	add(14, 14, 12)
	add(15, 15, 13)
	out(0, 14, 15)
	for k := 0; k < 2; k++ {
		b.VFMUL2D(16+k, 12+k, 25).VFMLA2D(16+k, 10+k, 24) // c51·t1 + c52·t3
		add(16+k, 0+k, 16+k)                              // r1
		b.VFMUL2D(18+k, 12+k, 24).VFMLA2D(18+k, 10+k, 25) // c52·t1 + c51·t3
		add(18+k, 0+k, 18+k)                              // r2
		b.VFMUL2D(20+k, 4+k, 27).VFMLA2D(20+k, 2+k, 26)   // u = s51·t2 + s52·t4
		b.VFMUL2D(22+k, 2+k, 27).VFMLS2D(22+k, 4+k, 26)   // v = s52·t2 - s51·t4
	}
	aMinusSz(0, 16, 21) // y1 = r1 + rotS(u)
	aPlusSz(1, 17, 20)
	out(1, 0, 1)
	aMinusSz(0, 18, 23) // y2 = r2 + rotS(v)
	aPlusSz(1, 19, 22)
	out(2, 0, 1)
	aPlusSz(0, 18, 23) // y3 = r2 - rotS(v)
	aMinusSz(1, 19, 22)
	out(3, 0, 1)
	aPlusSz(0, 16, 21) // y4 = r1 - rotS(u)
	aMinusSz(1, 17, 20)
	out(4, 0, 1)
}

// bfly8 is stockham.go's bfly8 as gc compiles it in pass8 and pass8last:
// beyond the ±s·z fusions, the ∓45°/∓135° rotations are computed unscaled,
// P = a5r - s·a5i, Q = a5i + s·a5r, R = -a7r - s·a7i (FNMADDD), T = -a7i +
// s·a7r (FNMSUBD), and their scaling by h = √2/2 is fused into the final add
// or subtract: y1 = a4 + h·(P, Q), y5 = a4 - h·(P, Q), y3 = a6 + h·(R, T),
// y7 = a6 - h·(R, T), each one FMADDD/FMSUBD. The kernel does the same with
// VFMLA/VFMLS on a copy of a4 and a6. h is in V18.
func bfly8(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := sops(b, inverse)
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
	sub(16, 16, 28) // a3 = a1 - a3, rotated where it is used
	sub(17, 17, 29)
	aMinusSz(22, 2, 7) // a5 = a5 + rotS(a7)
	aPlusSz(23, 3, 6)
	aPlusSz(24, 2, 7) // a7 = a5 - rotS(a7)
	aMinusSz(25, 3, 6)
	aMinusSz(2, 22, 23) // P
	aPlusSz(3, 23, 22)  // Q
	if inverse {
		b.VFNEG2D(6, 24)
		sub(6, 6, 25)  // R = -a7r - a7i
		sub(7, 24, 25) // T = -a7i + a7r
	} else {
		sub(6, 25, 24) // R = -a7r + a7i
		b.VFNEG2D(7, 25)
		sub(7, 7, 24) // T = -a7i - a7r
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
	out(0, 12, 13)
	b.Raw("VMOV V8.B16, V14.B16").Raw("VMOV V9.B16, V15.B16") // y1 = a4 + h·(P, Q)
	b.VFMLA2D(14, 2, 18).VFMLA2D(15, 3, 18)
	out(1, 14, 15)
	aMinusSz(14, 22, 17) // y2 = a2 + rotS(a3)
	aPlusSz(15, 23, 16)
	out(2, 14, 15)
	b.Raw("VMOV V10.B16, V14.B16").Raw("VMOV V11.B16, V15.B16") // y3 = a6 + h·(R, T)
	b.VFMLA2D(14, 6, 18).VFMLA2D(15, 7, 18)
	out(3, 14, 15)
	sub(14, 26, 20) // y4 = a0 - a1
	sub(15, 27, 21)
	out(4, 14, 15)
	b.VFMLS2D(8, 2, 18).VFMLS2D(9, 3, 18) // y5 = a4 - h·(P, Q), in place
	out(5, 8, 9)
	aPlusSz(14, 22, 17) // y6 = a2 - rotS(a3)
	aMinusSz(15, 23, 16)
	out(6, 14, 15)
	b.VFMLS2D(10, 6, 18).VFMLS2D(11, 7, 18) // y7 = a6 - h·(R, T), in place
	out(7, 10, 11)
}

// passStep emits one step of a radix-r pass: load the point(s), run the
// butterfly, multiply outputs 1..r-1 by their twiddles (V28/V29 ← w, V30/V31
// ← y·w) and store.
func passStep(b *arm64.Builder, r int, in, out []string, inverse bool, st step) {
	for j := 0; j < r; j++ {
		st.load(b, in[j], 2*j)
	}
	bfly(b, r, inverse, func(j, yr, yi int) {
		if j == 0 {
			st.store(b, yr, out[0])
			return
		}
		b.Raw("VLD1.P 32(R14), [V28.D2, V29.D2]")
		b.VFMUL2D(30, yr, 28).VFMLS2D(30, yi, 29) // re = yr·wr - yi·wi, yi·wi fused
		if r == 2 {
			b.VFMUL2D(31, yr, 29).VFMLA2D(31, yi, 28) // im = yr·wi + yi·wr, yi·wr fused
		} else {
			b.VFMUL2D(31, yi, 28).VFMLA2D(31, yr, 29) // im = yi·wr + yr·wi, yr·wi fused
		}
		if st == firstPair {
			b.Raw("VMOV V%d.D[0], V30.D[0]", yr).Raw("VMOV V%d.D[0], V31.D[0]", yi)
		}
		st.store(b, 30, out[j])
	})
}

// genStockhamPass emits skPass<r>NEON / skPass<r>NEONInv(cc, ch, tw
// *complex128, ido, l1 int): the radix-r pass for ido >= 2.
func genStockhamPass(f *emit.File, r int, inverse bool) {
	name := fmt.Sprintf("skPass%dNEON", r)
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "tw", "ido", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Int64}, nil, nil,
	)
	in, out := skStreams[r][0], skStreams[r][1]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R19").LoadArg("ch", "R20").LoadArg("tw", "R2").
		LoadArg("ido", "R3").LoadArg("l1", "R5").
		Raw("LSL $4, R3, R16"). // S: one input stream
		Raw("MUL R5, R16, R17") // OS: one output stream
	next := ""
	switch r {
	case 2:
		next = "ADD R16<<1, R19, R19"
	case 4:
		next = "ADD R16<<2, R19, R19"
	case 8:
		next = "ADD R16<<3, R19, R19"
	default:
		b.Raw("MOVD $%d, R4", r).Raw("MUL R4, R16, R4")
		next = "ADD R4, R19, R19"
	}
	skConsts(b, r)
	b.Label("kloop").Raw("MOVD R19, %s", in[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R16, %s, %s", in[j-1], in[j])
	}
	b.Raw("MOVD R20, %s", out[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("MOVD R2, R14")
	passStep(b, r, in, out, inverse, firstPair)
	b.Raw("LSR $1, R3, R15").
		Raw("SUBS $1, R15, R15").
		Raw("BEQ pairsdone")
	b.Label("iloop")
	passStep(b, r, in, out, inverse, nextPair)
	b.Raw("SUBS $1, R15, R15").
		Raw("BNE iloop")
	b.Label("pairsdone").
		Raw("TBZ $0, R3, knext") // odd ido: one point left
	passStep(b, r, in, out, inverse, lastPoint)
	b.Label("knext").
		Raw(next).
		Raw("ADD R16, R20, R20").
		Raw("SUBS $1, R5, R5").
		Raw("BNE kloop").
		Ret()
	f.Add(b.Func())
}

// genStockhamLast emits skLast<r>NEON / skLast<r>NEONInv(cc, ch *complex128,
// l1 int): the radix-r final pass (ido == 1) for an even l1, two blocks per
// iteration. Block k is the r points cc[r·k .. r·k+r-1]; two blocks are loaded
// as they lie (VLD1 into V(t), t = lastTemps[r], 2r registers), point j of
// blocks k and k+1 paired with VZIP1/VZIP2 into V(2j) (real parts) and
// V(2j+1) (imaginary parts), and output j of both blocks, ch[j·l1+k] and
// ch[j·l1+k+1], is one VST2. No twiddles.
func genStockhamLast(f *emit.File, r int, inverse bool) {
	name := fmt.Sprintf("skLast%dNEON", r)
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64}, nil, nil,
	)
	out := skStreams[r][1]
	t := map[int]int{2: 20, 3: 10, 4: 20, 5: 10}[r]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R0").LoadArg("ch", out[0]).LoadArg("l1", "R4").
		Raw("LSL $4, R4, R17")
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("LSR $1, R4, R5")
	skConsts(b, r)
	b.Label("loop")
	for v := 0; v < 2*r; {
		switch {
		case 2*r-v >= 4:
			b.Raw("VLD1.P 64(R0), [V%d.D2, V%d.D2, V%d.D2, V%d.D2]", t+v, t+v+1, t+v+2, t+v+3)
			v += 4
		default:
			b.Raw("VLD1.P 32(R0), [V%d.D2, V%d.D2]", t+v, t+v+1)
			v += 2
		}
	}
	for j := 0; j < r; j++ {
		b.Raw("VZIP1 V%d.D2, V%d.D2, V%d.D2", t+r+j, t+j, 2*j).
			Raw("VZIP2 V%d.D2, V%d.D2, V%d.D2", t+r+j, t+j, 2*j+1)
	}
	bfly(b, r, inverse, func(j, yr, yi int) {
		b.Raw("VST2.P [V%d.D2, V%d.D2], 32(%s)", yr, yi, out[j])
	})
	b.Raw("SUBS $1, R5, R5").
		Raw("BNE loop").
		Ret()
	f.Add(b.Func())
}

// genStockhamBatchNEON emits skBatch<r>NEON / skBatch<r>NEONInv(cc, ch, tw
// *complex128, ido, l1, pairs, odd, jin, jout, adjin, adjout int): one
// Stockham pass of radix r over a batch of w = 2·pairs + odd transforms laid
// side by side, the strips of a non-contiguous N-D axis (stockham.go's
// passBatch). Point p of the batch is w neighbouring complex128 values, so
// the pair of lines c, c+1 at one point is a VLD2 as in a 1-D pass, and every
// lane runs exactly the arithmetic the 1-D kernels (and the Go passes) run on
// one line: the same butterflies, the same fused twiddle product. A point's
// twiddle is the same for every line, so it is loaded once per pair already
// broadcast and split, re into V28 and im into V29, by VLD2R from the batched
// twiddle table (batchTwiddles: for i = 1 .. ido-1, the r-1 twiddles of point
// i). Point i = 0 is not multiplied at all, as in the Go pass, so no lane needs
// restoring. An odd batch ends with one line alone, loaded into lane 0 with
// FLDPD as the 1-D kernels do with the last point of an odd ido.
//
// Input stream j of block k starts at cc + k·r·jin + j·jin, output stream j at
// ch + k·ido·sOut·16 + j·jout; each row (one point i) advances its streams by
// 32·pairs through the post-indexed loads and stores and then by adjin and
// adjout, which the caller sets to 16·(stride - 2·pairs), to reach the next
// point's row. After its ido rows, stream 0 of the output has reached the next
// block's output, and stream 0 of the input is one stream (jin) past its block
// start, so it skips r-1 more.
//
// Registers: the streams as in the pass kernels (skStreams), R2 the table, R26
// the current point's twiddles and R14 their cursor, R3 the rows left, R5 the
// blocks left, R15 the pairs left, R16 jin, R17 jout, R19 and R20 scratch.
func genStockhamBatchNEON(f *emit.File, r int, inverse bool) {
	name := fmt.Sprintf("skBatch%dNEON", r)
	if inverse {
		name += "Inv"
	}
	names := []string{"cc", "ch", "tw", "ido", "l1", "pairs", "odd", "jin", "jout", "adjin", "adjout"}
	types := []arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr}
	for range names[3:] {
		types = append(types, arm64.Int64)
	}
	sig := arm64.Layout(names, types, nil, nil)
	in, out := skStreams[r][0], skStreams[r][1]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", in[0]).LoadArg("ch", out[0]).LoadArg("tw", "R2").
		LoadArg("l1", "R5").LoadArg("jin", "R16").LoadArg("jout", "R17")
	skConsts(b, r)
	// row emits one point's row of the batch: the pairs, the odd line, then
	// the step to the next point's row.
	row := func(label string, twiddled bool) {
		b.LoadArg("pairs", "R15").
			Raw("CBZ R15, %so", label)
		b.Label(label + "p")
		batchStepNEON(b, r, in, out, inverse, twiddled, false)
		b.Raw("SUBS $1, R15, R15").
			Raw("BNE %sp", label)
		b.Label(label+"o").
			LoadArg("odd", "R19").
			Raw("CBZ R19, %se", label)
		batchStepNEON(b, r, in, out, inverse, twiddled, true)
		b.Label(label+"e").
			LoadArg("adjin", "R19").
			LoadArg("adjout", "R20")
		for j := 0; j < r; j++ {
			b.Raw("ADD R19, %s, %s", in[j], in[j]).
				Raw("ADD R20, %s, %s", out[j], out[j])
		}
	}
	b.Label("kloop")
	for j := 1; j < r; j++ {
		b.Raw("ADD R16, %s, %s", in[j-1], in[j]).
			Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	row("zero", false) // point i = 0: no twiddle
	b.LoadArg("ido", "R3").
		Raw("SUBS $1, R3, R3").
		Raw("BEQ knext").
		Raw("MOVD R2, R26")
	b.Label("iloop")
	row("tw", true)
	b.Raw("ADD $%d, R26, R26", (r-1)*16).
		Raw("SUBS $1, R3, R3").
		Raw("BNE iloop")
	b.Label("knext")
	switch r { // input stream 0: r-1 more streams to the next block
	case 2:
		b.Raw("ADD R16, %s, %s", in[0], in[0])
	case 3:
		b.Raw("ADD R16<<1, %s, %s", in[0], in[0])
	case 4:
		b.Raw("ADD R16<<1, %s, %s", in[0], in[0]).Raw("ADD R16, %s, %s", in[0], in[0])
	case 5:
		b.Raw("ADD R16<<2, %s, %s", in[0], in[0])
	case 8:
		b.Raw("ADD R16<<3, %s, %s", in[0], in[0]).Raw("SUB R16, %s, %s", in[0], in[0])
	}
	b.Raw("SUBS $1, R5, R5").
		Raw("BNE kloop").
		Ret()
	f.Add(b.Func())
}

// batchStepNEON emits one step of a batched pass: a pair of lines (or, with
// single, the last line alone) at one point: load, butterfly, and, when
// twiddled, outputs 1..r-1 multiplied by the point's twiddles with the 1-D
// kernels' fused product (see passStep), then store.
func batchStepNEON(b *arm64.Builder, r int, in, out []string, inverse, twiddled, single bool) {
	for j := 0; j < r; j++ {
		if single {
			b.Raw("FLDPD (%s), (F%d, F%d)", in[j], 2*j, 2*j+1)
		} else {
			b.Raw("VLD2.P 32(%s), [V%d.D2, V%d.D2]", in[j], 2*j, 2*j+1)
		}
	}
	if twiddled {
		b.Raw("MOVD R26, R14")
	}
	store := func(v int, ptr string) {
		if single {
			b.Raw("FSTPD (F%d, F%d), (%s)", v, v+1, ptr)
		} else {
			b.Raw("VST2.P [V%d.D2, V%d.D2], 32(%s)", v, v+1, ptr)
		}
	}
	bfly(b, r, inverse, func(j, yr, yi int) {
		if j == 0 || !twiddled {
			store(yr, out[j])
			return
		}
		b.Raw("VLD2R.P 16(R14), [V28.D2, V29.D2]")
		b.VFMUL2D(30, yr, 28).VFMLS2D(30, yi, 29) // re = yr·wr - yi·wi, yi·wi fused
		if r == 2 {
			b.VFMUL2D(31, yr, 29).VFMLA2D(31, yi, 28) // im = yr·wi + yi·wr, yi·wr fused
		} else {
			b.VFMUL2D(31, yi, 28).VFMLA2D(31, yr, 29) // im = yi·wr + yr·wi, yr·wi fused
		}
		store(30, out[j])
	})
}

// The split-layout kernels keep a transform's data block-split between its
// passes, as KFR does (splitin/splitout), so the passes between the first and
// the last load and store with VLD1/VST1 instead of VLD2/VST2. In the
// block-split layout the pair of points 2q, 2q+1 occupies the same 32 bytes as
// when interleaved, ordered re(2q), re(2q+1), im(2q), im(2q+1): one VLD1 of two
// registers is then exactly what a VLD2 makes of the interleaved pair, with no
// deinterleave, and every address is unchanged. The first pass of a transform
// reads its input interleaved and writes split (out), the passes between read
// and write split (in, out), and the last reads split and writes interleaved
// (genStockhamLastSplit), so the caller sees only interleaved data. The
// arithmetic is that of the interleaved kernels, instruction for instruction,
// so the results are the same bits.
//
// A pair must not straddle two streams, so every stream must start at an even
// point: the passes run with an even ido only, and the final pass with an
// even radix (2 or 4) and an even l1. That leaves the powers of two.

// splitLayout says which side of a pass is block-split.
type splitLayout struct{ in, out bool }

// suffix names the kernel: IS reads interleaved and writes split, SS both
// split.
func (l splitLayout) suffix() string {
	if l.in {
		return "SS"
	}
	return "IS"
}

// splitPassStep is passStep for a pair of points (ido is even, so there is
// never a lone point) with the loads and stores of the layout.
func splitPassStep(b *arm64.Builder, r int, in, out []string, inverse bool, first bool, lay splitLayout) {
	for j := 0; j < r; j++ {
		if lay.in {
			b.Raw("VLD1.P 32(%s), [V%d.D2, V%d.D2]", in[j], 2*j, 2*j+1)
		} else {
			b.Raw("VLD2.P 32(%s), [V%d.D2, V%d.D2]", in[j], 2*j, 2*j+1)
		}
	}
	store := func(v int, ptr string) {
		if lay.out {
			b.Raw("VST1.P [V%d.D2, V%d.D2], 32(%s)", v, v+1, ptr)
		} else {
			b.Raw("VST2.P [V%d.D2, V%d.D2], 32(%s)", v, v+1, ptr)
		}
	}
	bfly(b, r, inverse, func(j, yr, yi int) {
		if j == 0 {
			store(yr, out[0])
			return
		}
		b.Raw("VLD1.P 32(R14), [V28.D2, V29.D2]")
		b.VFMUL2D(30, yr, 28).VFMLS2D(30, yi, 29) // re = yr·wr - yi·wi, yi·wi fused
		if r == 2 {
			b.VFMUL2D(31, yr, 29).VFMLA2D(31, yi, 28) // im = yr·wi + yi·wr, yi·wr fused
		} else {
			b.VFMUL2D(31, yi, 28).VFMLA2D(31, yr, 29) // im = yi·wr + yr·wi, yr·wi fused
		}
		if first {
			b.Raw("VMOV V%d.D[0], V30.D[0]", yr).Raw("VMOV V%d.D[0], V31.D[0]", yi)
		}
		store(30, out[j])
	})
}

// genStockhamPassSplit emits skPass<r>NEON<IS|SS> and the Inv forms (cc, ch,
// tw *complex128, ido, l1 int): genStockhamPass's radix-r pass, for an even
// ido, with the layout's loads and stores.
func genStockhamPassSplit(f *emit.File, r int, inverse bool, lay splitLayout) {
	name := fmt.Sprintf("skPass%dNEON", r)
	if inverse {
		name += "Inv"
	}
	name += lay.suffix()
	sig := arm64.Layout(
		[]string{"cc", "ch", "tw", "ido", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Int64}, nil, nil,
	)
	in, out := skStreams[r][0], skStreams[r][1]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R19").LoadArg("ch", "R20").LoadArg("tw", "R2").
		LoadArg("ido", "R3").LoadArg("l1", "R5").
		Raw("LSL $4, R3, R16"). // one input stream
		Raw("MUL R5, R16, R17") // one output stream
	next := map[int]string{2: "ADD R16<<1, R19, R19", 4: "ADD R16<<2, R19, R19", 8: "ADD R16<<3, R19, R19"}[r]
	skConsts(b, r)
	b.Label("kloop").Raw("MOVD R19, %s", in[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R16, %s, %s", in[j-1], in[j])
	}
	b.Raw("MOVD R20, %s", out[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("MOVD R2, R14")
	splitPassStep(b, r, in, out, inverse, true, lay)
	b.Raw("LSR $1, R3, R15").
		Raw("SUBS $1, R15, R15").
		Raw("BEQ knext")
	b.Label("iloop")
	splitPassStep(b, r, in, out, inverse, false, lay)
	b.Raw("SUBS $1, R15, R15").
		Raw("BNE iloop")
	b.Label("knext").
		Raw(next).
		Raw("ADD R16, R20, R20").
		Raw("SUBS $1, R5, R5").
		Raw("BNE kloop").
		Ret()
	f.Add(b.Func())
}

// genStockhamLastSplit emits skLast<r>NEONSI / skLast<r>NEONInvSI(cc, ch
// *complex128, l1 int): genStockhamLast's final pass, reading block-split
// input. Blocks k and k+1 are the 2r points from r·k, r pairs loaded as they
// lie into V(t+2q) (real parts of pair q) and V(t+2q+1) (imaginary parts).
// Point j of block k is lane j%2 of pair j/2, and of block k+1 the same lane of
// pair r/2+j/2 (r is even), so VZIP1 (j even) or VZIP2 (j odd) of the two
// pairs puts point j of both blocks in V(2j) and V(2j+1), as the interleaved
// kernel's zips do.
func genStockhamLastSplit(f *emit.File, r int, inverse bool) {
	name := fmt.Sprintf("skLast%dNEON", r)
	if inverse {
		name += "Inv"
	}
	name += "SI"
	sig := arm64.Layout(
		[]string{"cc", "ch", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64}, nil, nil,
	)
	out := skStreams[r][1]
	t := 20
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R0").LoadArg("ch", out[0]).LoadArg("l1", "R4").
		Raw("LSL $4, R4, R17")
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("LSR $1, R4, R5")
	skConsts(b, r)
	b.Label("loop")
	for v := 0; v < 2*r; v += 4 {
		b.Raw("VLD1.P 64(R0), [V%d.D2, V%d.D2, V%d.D2, V%d.D2]", t+v, t+v+1, t+v+2, t+v+3)
	}
	for j := 0; j < r; j++ {
		qa, qb := j/2, r/2+j/2
		z := "VZIP1"
		if j%2 == 1 {
			z = "VZIP2"
		}
		b.Raw("%s V%d.D2, V%d.D2, V%d.D2", z, t+2*qb, t+2*qa, 2*j).
			Raw("%s V%d.D2, V%d.D2, V%d.D2", z, t+2*qb+1, t+2*qa+1, 2*j+1)
	}
	bfly(b, r, inverse, func(j, yr, yi int) {
		b.Raw("VST2.P [V%d.D2, V%d.D2], 32(%s)", yr, yi, out[j])
	})
	b.Raw("SUBS $1, R5, R5").
		Raw("BNE loop").
		Ret()
	f.Add(b.Func())
}

// ---------------------------------------------------------------------------
// float32 (complex64) Stockham pass kernels, into stockham32_arm64.s.
//
// sk32Pass<r>NEON[Inv](cc, ch, tw *complex64, ido, l1 int) and
// sk32Last<r>NEON[Inv](cc, ch *complex64, l1 int) run the fft package's
// float32 passes (stockham32_passes.go, f32Pass<r> and f32Pass<r>last) for
// radix 2, 3, 4, 5 and 8, four points per register: VLD2 splits four
// complex64 into a register of real parts and one of imaginary parts (.4S),
// every lane then runs the Go pass's operation for one point, and VST2
// interleaves on store. They are the float64 kernels above with
// single-precision registers, in their own functions (f32simd*) so a change
// to one family cannot move the other's arithmetic.
//
// Bit identity. gc compiles the float32 passes with the same fused
// multiply-adds as the float64 ones (FMADDS/FMSUBS where those have
// FMADDD/FMSUBD, go build -gcflags=-S, Go 1.27.1: pass2, pass3, pass4 and
// every final pass emit the same sequence of operations; pass5, pass7 and
// pass8 the same operations scheduled differently), so the kernels fuse the
// products the float64 kernels fuse: see the comment above step. The tests
// compare kernel and Go pass bit for bit.
//
// Points are walked in groups of four (one .4S register pair), then a group
// of two (VLD2/VST2 on .2S, the upper lanes zero) for ido&2 and a lone point
// (FLDPS/FSTPS) for ido&1. Twiddles come in the same groups, one sequential
// stream already split (kernels.StockhamTwiddles32 on arm64): for a group of
// g points and each j, the g real parts then the g imaginary parts, loaded
// with VLD1 (FLDPS for a lone point). The point i = 0 is multiplied by w = 1
// with its group and lane 0 of the untwiddled value put back (VMOV element).
//
// Registers as in the float64 kernels, with S = ido·8 and OS = l1·ido·8.

// f32simdStep is one call of a float32 pass body: g points (4, 2 or 1), the
// first group of a block (whose lane 0 is restored) or another.
type f32simdStep struct {
	g     int
	first bool
}

func (st f32simdStep) load(b *arm64.Builder, ptr string, v int) {
	switch st.g {
	case 4:
		b.Raw("VLD2.P 32(%s), [V%d.S4, V%d.S4]", ptr, v, v+1)
	case 2:
		b.Raw("VLD2.P 16(%s), [V%d.S2, V%d.S2]", ptr, v, v+1)
	default:
		b.Raw("FLDPS (%s), (F%d, F%d)", ptr, v, v+1)
	}
}

func (st f32simdStep) store(b *arm64.Builder, v int, ptr string) {
	switch st.g {
	case 4:
		b.Raw("VST2.P [V%d.S4, V%d.S4], 32(%s)", v, v+1, ptr)
	case 2:
		b.Raw("VST2.P [V%d.S2, V%d.S2], 16(%s)", v, v+1, ptr)
	default:
		b.Raw("FSTPS (F%d, F%d), (%s)", v, v+1, ptr)
	}
}

// twLoad loads the next twiddle of the group into V28 (real parts) and V29
// (imaginary parts), and moves R14 past it.
func (st f32simdStep) twLoad(b *arm64.Builder) {
	switch st.g {
	case 4:
		b.Raw("VLD1.P 32(R14), [V28.S4, V29.S4]")
	case 2:
		b.Raw("VLD1.P 16(R14), [V28.S2, V29.S2]")
	default:
		b.Raw("FLDPS (R14), (F28, F29)").Raw("ADD $8, R14")
	}
}

// f32simdConsts is skConsts in float32: each constant rounded once to float32
// (cmd/asm rounds the decimal to float64 and then to float32; for these
// constants that equals rounding the exact value once, which the bit-exact
// tests confirm) and duplicated into four lanes. h is the register for √2/2.
func f32simdConsts(b *arm64.Builder, r, h int) {
	dup := func(v int, c string) {
		b.Raw("FMOVS $(%s), F%d", c, v).Raw("VDUP V%d.S[0], V%d.S4", v, v)
	}
	switch r {
	case 3:
		dup(24, "0.5")
		dup(25, "0.8660254037844386467637231707529361834714026269051903140279") // sin(2π/3)
	case 5:
		dup(24, "0.30901699437494742410229341718281905886015458990288")  // c51 = cos(2π/5)
		dup(25, "-0.80901699437494742410229341718281905886015458990289") // c52 = cos(4π/5)
		dup(26, "0.95105651629515357211643933337938214340569863412575")  // s51 = sin(2π/5)
		dup(27, "0.58778525229247312916870595463907276859765243764314")  // s52 = sin(4π/5)
	case 8:
		dup(h, "0.707106781186547524400844362104849") // √2/2
	}
}

func f32simdBfly(b *arm64.Builder, r int, inverse bool, out func(j, yr, yi int)) {
	map[int]func(*arm64.Builder, bool, func(j, yr, yi int)){2: f32simdBfly2, 3: f32simdBfly3, 4: f32simdBfly4, 5: f32simdBfly5, 8: f32simdBfly8}[r](b, inverse, out)
}

// f32simdSops is sops on .4S.
func f32simdSops(b *arm64.Builder, inverse bool) (add, sub, aMinusSz, aPlusSz func(d, n, m int)) {
	add = func(d, n, m int) { b.VFADD4S(d, n, m) }
	sub = func(d, n, m int) { b.VFSUB4S(d, n, m) }
	pm := func(d, n, m int, plus bool) {
		if plus {
			add(d, n, m)
		} else {
			sub(d, n, m)
		}
	}
	aMinusSz = func(d, a, z int) { pm(d, a, z, !inverse) }
	aPlusSz = func(d, a, z int) { pm(d, a, z, inverse) }
	return
}

// f32simdBfly2 … f32simdBfly8 are bfly2 … bfly8 on .4S, operation for
// operation.
func f32simdBfly2(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	b.VFADD4S(4, 0, 2).VFADD4S(5, 1, 3)
	out(0, 4, 5)
	b.VFSUB4S(4, 0, 2).VFSUB4S(5, 1, 3)
	out(1, 4, 5)
}

func f32simdBfly3(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := f32simdSops(b, inverse)
	add(6, 2, 4) // t1
	add(7, 3, 5)
	sub(2, 2, 4) // t2
	sub(3, 3, 5)
	add(8, 0, 6) // y0
	add(9, 1, 7)
	out(0, 8, 9)
	b.VFMLS4S(0, 6, 24).VFMLS4S(1, 7, 24) // ca, in place on x0
	b.VFMUL4S(2, 2, 25).VFMUL4S(3, 3, 25) // q, in place on t2
	aMinusSz(10, 0, 3)                    // y1 = ca + rotS(q)
	aPlusSz(11, 1, 2)
	out(1, 10, 11)
	aPlusSz(10, 0, 3) // y2 = ca - rotS(q)
	aMinusSz(11, 1, 2)
	out(2, 10, 11)
}

func f32simdBfly4(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := f32simdSops(b, inverse)
	add(8, 0, 4) // t2
	add(9, 1, 5)
	sub(10, 0, 4) // t1
	sub(11, 1, 5)
	add(12, 2, 6) // t3
	add(13, 3, 7)
	sub(14, 2, 6) // t4
	sub(15, 3, 7)
	add(16, 8, 12) // y0
	add(17, 9, 13)
	out(0, 16, 17)
	aMinusSz(0, 10, 15) // y1 = t1 + rotS(t4)
	aPlusSz(1, 11, 14)
	out(1, 0, 1)
	sub(16, 8, 12) // y2
	sub(17, 9, 13)
	out(2, 16, 17)
	aPlusSz(0, 10, 15) // y3 = t1 - rotS(t4)
	aMinusSz(1, 11, 14)
	out(3, 0, 1)
}

func f32simdBfly5(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := f32simdSops(b, inverse)
	add(10, 2, 8) // t1
	add(11, 3, 9)
	sub(2, 2, 8) // t2
	sub(3, 3, 9)
	add(12, 4, 6) // t3
	add(13, 5, 7)
	sub(4, 4, 6) // t4
	sub(5, 5, 7)
	add(14, 0, 10) // y0 = (a + t1) + t3
	add(15, 1, 11)
	add(14, 14, 12)
	add(15, 15, 13)
	out(0, 14, 15)
	for k := 0; k < 2; k++ {
		b.VFMUL4S(16+k, 12+k, 25).VFMLA4S(16+k, 10+k, 24) // c51·t1 + c52·t3
		add(16+k, 0+k, 16+k)                              // r1
		b.VFMUL4S(18+k, 12+k, 24).VFMLA4S(18+k, 10+k, 25) // c52·t1 + c51·t3
		add(18+k, 0+k, 18+k)                              // r2
		b.VFMUL4S(20+k, 4+k, 27).VFMLA4S(20+k, 2+k, 26)   // u = s51·t2 + s52·t4
		b.VFMUL4S(22+k, 2+k, 27).VFMLS4S(22+k, 4+k, 26)   // v = s52·t2 - s51·t4
	}
	aMinusSz(0, 16, 21) // y1 = r1 + rotS(u)
	aPlusSz(1, 17, 20)
	out(1, 0, 1)
	aMinusSz(0, 18, 23) // y2 = r2 + rotS(v)
	aPlusSz(1, 19, 22)
	out(2, 0, 1)
	aPlusSz(0, 18, 23) // y3 = r2 - rotS(v)
	aMinusSz(1, 19, 22)
	out(3, 0, 1)
	aPlusSz(0, 16, 21) // y4 = r1 - rotS(u)
	aMinusSz(1, 17, 20)
	out(4, 0, 1)
}

// f32simdBfly8 is bfly8; √2/2 is in V18.
func f32simdBfly8(b *arm64.Builder, inverse bool, out func(j, yr, yi int)) {
	add, sub, aMinusSz, aPlusSz := f32simdSops(b, inverse)
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
	sub(16, 16, 28) // a3 = a1 - a3, rotated where it is used
	sub(17, 17, 29)
	aMinusSz(22, 2, 7) // a5 = a5 + rotS(a7)
	aPlusSz(23, 3, 6)
	aPlusSz(24, 2, 7) // a7 = a5 - rotS(a7)
	aMinusSz(25, 3, 6)
	aMinusSz(2, 22, 23) // P
	aPlusSz(3, 23, 22)  // Q
	if inverse {
		b.VFNEG4S(6, 24)
		sub(6, 6, 25)  // R = -a7r - a7i
		sub(7, 24, 25) // T = -a7i + a7r
	} else {
		sub(6, 25, 24) // R = -a7r + a7i
		b.VFNEG4S(7, 25)
		sub(7, 7, 24) // T = -a7i - a7r
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
	out(0, 12, 13)
	b.Raw("VMOV V8.B16, V14.B16").Raw("VMOV V9.B16, V15.B16") // y1 = a4 + h·(P, Q)
	b.VFMLA4S(14, 2, 18).VFMLA4S(15, 3, 18)
	out(1, 14, 15)
	aMinusSz(14, 22, 17) // y2 = a2 + rotS(a3)
	aPlusSz(15, 23, 16)
	out(2, 14, 15)
	b.Raw("VMOV V10.B16, V14.B16").Raw("VMOV V11.B16, V15.B16") // y3 = a6 + h·(R, T)
	b.VFMLA4S(14, 6, 18).VFMLA4S(15, 7, 18)
	out(3, 14, 15)
	sub(14, 26, 20) // y4 = a0 - a1
	sub(15, 27, 21)
	out(4, 14, 15)
	b.VFMLS4S(8, 2, 18).VFMLS4S(9, 3, 18) // y5 = a4 - h·(P, Q), in place
	out(5, 8, 9)
	aPlusSz(14, 22, 17) // y6 = a2 - rotS(a3)
	aMinusSz(15, 23, 16)
	out(6, 14, 15)
	b.VFMLS4S(10, 6, 18).VFMLS4S(11, 7, 18) // y7 = a6 - h·(R, T), in place
	out(7, 10, 11)
}

// f32simdPassStep emits one step of a float32 radix-r pass: passStep with
// the fusions of the twiddle product as gc compiles f32Mul in each pass.
func f32simdPassStep(b *arm64.Builder, r int, in, out []string, inverse bool, st f32simdStep) {
	for j := 0; j < r; j++ {
		st.load(b, in[j], 2*j)
	}
	f32simdBfly(b, r, inverse, func(j, yr, yi int) {
		if j == 0 {
			st.store(b, yr, out[0])
			return
		}
		st.twLoad(b)
		b.VFMUL4S(30, yr, 28).VFMLS4S(30, yi, 29) // re = yr·wr - yi·wi, yi·wi fused
		if r == 2 {
			b.VFMUL4S(31, yr, 29).VFMLA4S(31, yi, 28) // im = yr·wi + yi·wr, yi·wr fused
		} else {
			b.VFMUL4S(31, yi, 28).VFMLA4S(31, yr, 29) // im = yi·wr + yr·wi, yr·wi fused
		}
		if st.first {
			b.Raw("VMOV V%d.S[0], V30.S[0]", yr).Raw("VMOV V%d.S[0], V31.S[0]", yi)
		}
		st.store(b, 30, out[j])
	})
}

// genF32StockhamPass emits sk32Pass<r>NEON / sk32Pass<r>NEONInv (ido >= 2).
func genF32StockhamPass(f *emit.File, r int, inverse bool) {
	name := fmt.Sprintf("sk32Pass%dNEON", r)
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "tw", "ido", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Int64}, nil, nil,
	)
	in, out := skStreams[r][0], skStreams[r][1]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R19").LoadArg("ch", "R20").LoadArg("tw", "R2").
		LoadArg("ido", "R3").LoadArg("l1", "R5").
		Raw("LSL $3, R3, R16"). // S: one input stream
		Raw("MUL R5, R16, R17") // OS: one output stream
	next := ""
	switch r {
	case 2:
		next = "ADD R16<<1, R19, R19"
	case 4:
		next = "ADD R16<<2, R19, R19"
	case 8:
		next = "ADD R16<<3, R19, R19"
	default:
		b.Raw("MOVD $%d, R4", r).Raw("MUL R4, R16, R4")
		next = "ADD R4, R19, R19"
	}
	f32simdConsts(b, r, 18)
	b.Label("kloop").Raw("MOVD R19, %s", in[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R16, %s, %s", in[j-1], in[j])
	}
	b.Raw("MOVD R20, %s", out[0])
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("MOVD R2, R14").
		Raw("LSR $2, R3, R15"). // groups of four points
		Raw("CBZ R15, firstpair")
	f32simdPassStep(b, r, in, out, inverse, f32simdStep{g: 4, first: true})
	b.Raw("SUBS $1, R15, R15").
		Raw("BEQ quadsdone")
	b.Label("qloop")
	f32simdPassStep(b, r, in, out, inverse, f32simdStep{g: 4})
	b.Raw("SUBS $1, R15, R15").
		Raw("BNE qloop")
	b.Label("quadsdone").
		Raw("TBZ $1, R3, single") // ido&2: a group of two
	f32simdPassStep(b, r, in, out, inverse, f32simdStep{g: 2})
	b.Raw("B single")
	b.Label("firstpair") // ido is 2 or 3: the first group is the pair
	f32simdPassStep(b, r, in, out, inverse, f32simdStep{g: 2, first: true})
	b.Label("single").
		Raw("TBZ $0, R3, knext") // ido&1: one point left
	f32simdPassStep(b, r, in, out, inverse, f32simdStep{g: 1})
	b.Label("knext").
		Raw(next).
		Raw("ADD R16, R20, R20").
		Raw("SUBS $1, R5, R5").
		Raw("BNE kloop").
		Ret()
	f.Add(b.Func())
}

// f32simdUzp splits the pairs of complex64 p (blocks k, k+1) and q (blocks
// k+2, k+3) into the real parts of the four blocks, V(d), and their
// imaginary parts, V(d+1). d may be p (the imaginary parts are taken first);
// d+1 must be neither p nor q.
func f32simdUzp(b *arm64.Builder, p, q, d int) {
	b.Raw("VUZP2 V%d.S4, V%d.S4, V%d.S4", q, p, d+1).
		Raw("VUZP1 V%d.S4, V%d.S4, V%d.S4", q, p, d)
}

// f32simdLastLoad loads four blocks of r points (4r complex64 from R0) and
// leaves point j of the four blocks split in V(2j) (real parts) and V(2j+1)
// (imaginary parts). Viewed as 64-bit lanes, one register pair p, q holds
// point j of blocks k, k+1 and of blocks k+2, k+3, and VUZP1/VUZP2 split
// them. Radix 2, 3 and 4 get p and q straight from VLD2/VLD3/VLD4 on .2D
// lanes (one per block pair); radix 8 from VLD4 on one block at a time,
// whose points j and j+4 share a register, paired across blocks with
// VZIP1/VZIP2; radix 5, which has no structure load, from plain VLD1 loads
// (T0..T4 = points 0-1, 2-3, 4|0, 1-2, 3-4 of a block pair), with VEXT for
// the pairs that straddle two registers and an element move (VMOV D[1]) for
// the others.
func f32simdLastLoad(b *arm64.Builder, r int) {
	regs := func(first, n int) string {
		s := ""
		for i := 0; i < n; i++ {
			if i > 0 {
				s += ", "
			}
			s += fmt.Sprintf("V%d.D2", first+i)
		}
		return s
	}
	switch r {
	case 2, 3, 4:
		b.Raw("VLD%d.P %d(R0), [%s]", r, 16*r, regs(16, r)).
			Raw("VLD%d.P %d(R0), [%s]", r, 16*r, regs(20, r))
		for j := 0; j < r; j++ {
			f32simdUzp(b, 16+j, 20+j, 2*j)
		}
	case 8:
		for q := 0; q < 4; q++ {
			b.Raw("VLD4.P 64(R0), [%s]", regs(16+4*q, 4))
		}
		for i := 0; i < 4; i++ {
			b.Raw("VZIP1 V%d.D2, V%d.D2, V%d.D2", 20+i, 16+i, 2*i).
				Raw("VZIP2 V%d.D2, V%d.D2, V%d.D2", 20+i, 16+i, 8+2*i).
				Raw("VZIP1 V%d.D2, V%d.D2, V%d.D2", 28+i, 24+i, 16+i).
				Raw("VZIP2 V%d.D2, V%d.D2, V%d.D2", 28+i, 24+i, 20+i)
			f32simdUzp(b, 2*i, 16+i, 2*i)
			f32simdUzp(b, 8+2*i, 20+i, 8+2*i)
		}
	case 5:
		b.Raw("VLD1.P 64(R0), [%s]", regs(16, 4)).
			Raw("VLD1.P 16(R0), [V20.D2]").
			Raw("VLD1.P 32(R0), [V21.D2, V22.D2]").
			Raw("VLD1.P 16(R0), [V23.D2]").
			Raw("VLD1.P 32(R0), [V28.D2, V29.D2]")
		pairs := func(t [5]int, e1, e3 int) [5]int {
			b.Raw("VEXT $8, V%d.B16, V%d.B16, V%d.B16", t[3], t[0], e1). // points 1 | 1
											Raw("VEXT $8, V%d.B16, V%d.B16, V%d.B16", t[4], t[1], e3). // points 3 | 3
											Raw("VMOV V%d.D[1], V%d.D[1]", t[2], t[0]).                // points 0 | 0
											Raw("VMOV V%d.D[1], V%d.D[1]", t[3], t[1]).                // points 2 | 2
											Raw("VMOV V%d.D[1], V%d.D[1]", t[4], t[2])                 // points 4 | 4
			return [5]int{t[0], e1, t[1], e3, t[2]}
		}
		p := pairs([5]int{16, 17, 18, 19, 20}, 10, 11)
		q := pairs([5]int{21, 22, 23, 28, 29}, 12, 13)
		for j := 0; j < 5; j++ {
			f32simdUzp(b, p[j], q[j], 2*j)
		}
	}
}

// genF32StockhamLast emits sk32Last<r>NEON / sk32Last<r>NEONInv(cc, ch
// *complex64, l1 int): the final pass (ido == 1) for l1 a multiple of four,
// four blocks per iteration (f32simdLastLoad); output j of the four blocks,
// ch[j·l1+k .. j·l1+k+3], is one VST2.
func genF32StockhamLast(f *emit.File, r int, inverse bool) {
	name := fmt.Sprintf("sk32Last%dNEON", r)
	if inverse {
		name += "Inv"
	}
	sig := arm64.Layout(
		[]string{"cc", "ch", "l1"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64}, nil, nil,
	)
	out := skStreams[r][1]
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("cc", "R0").LoadArg("ch", out[0]).LoadArg("l1", "R4").
		Raw("LSL $3, R4, R17")
	for j := 1; j < r; j++ {
		b.Raw("ADD R17, %s, %s", out[j-1], out[j])
	}
	b.Raw("LSR $2, R4, R5")
	if r != 8 {
		f32simdConsts(b, r, 18)
	}
	b.Label("loop")
	f32simdLastLoad(b, r)
	if r == 8 {
		// √2/2 again: the loads above used V18.
		f32simdConsts(b, r, 18)
	}
	f32simdBfly(b, r, inverse, func(j, yr, yi int) {
		b.Raw("VST2.P [V%d.S4, V%d.S4], 32(%s)", yr, yi, out[j])
	})
	b.Raw("SUBS $1, R5, R5").
		Raw("BNE loop").
		Ret()
	f.Add(b.Func())
}

// genF32StockhamFile writes the float32 pass kernels.
func genF32StockhamFile() {
	f := emit.NewFile("arm64")
	for _, inverse := range []bool{false, true} {
		for _, r := range []int{2, 3, 4, 5, 8} {
			genF32StockhamPass(f, r, inverse)
		}
		for _, r := range []int{2, 3, 4, 5, 8} {
			genF32StockhamLast(f, r, inverse)
		}
	}
	write("stockham32_arm64.s", f)
}

// Command gen writes the amd64 kernels of the kernels package, via go-asmgen.
// Run with: go run gen.go (or `go generate` from the kernels package).
//
// The kernels themselves are built by go-asmgen's fft/amd64 package; this
// file says which kernels the package has, under which names, in which files,
// and where their constants live (the tables of stockham_amd64.go and
// stockhamsplit_amd64.go). Every kernel's arithmetic is a contract with a Go
// reference in this module, which the tests hold it to bit for bit.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
	fft "github.com/go-asmgen/asmgen/fft/amd64"
)

// k builds the kernels that read the constants tables: skConst (and its
// float32 and AVX-512 forms, whose rows sit at the same offsets, doubled at
// 512 bits) and splitConst.
var k = &fft.Kernels{
	K: fft.Consts{
		Sign: 0, Half: 32, Sin120: 64, C51: 96, C52: 128, S51: 160, S52: 192, H: 224,
		AddSub: 256,
		R16C1:  288, R16S1: 320, R16NH: 352, R16NC1: 384,
		R16SS1: 416, R16SSH: 448, R16SSC1: 480, R16SSN1: 512,
		R9C1: 544, R9S1: 576, R9C2: 608, R9S2: 640, R9C4: 672, R9S4: 704,
		R32: 736,
	},
	SplitK: fft.SplitConsts{H: 0, Neg: 32, Half: 64, Sin120: 96, C51: 128, C52: 160, S51: 192, S52: 224, R32: 256},
}

// kf builds the FMA twins (Round 41): k's kernels with the products fused.
var kf = &fft.Kernels{K: k.K, SplitK: k.SplitK, FMA: true}

func main() {
	fc := emit.NewFile("amd64")
	fft.CMulSSE2(fc, "cmulSSE2")
	writeFile("cmul_amd64.s", fc.String())

	fb := emit.NewFile("amd64")
	fft.Radix2SSE2(fb, "radix2StageSSE2")
	fft.Radix4SSE2(fb, "radix4StageSSE2Fwd", false)
	fft.Radix4SSE2(fb, "radix4StageSSE2Inv", true)
	for _, r := range []int{2, 3, 4, 5, 8} {
		k.Pass(fb, fmt.Sprintf("skPass%dAVX2", r), r)
	}
	for _, r := range []int{2, 3, 4, 5, 8} {
		k.Last(fb, fmt.Sprintf("skLast%dAVX2", r), r)
	}
	// AVX-512 only for the radices of a power-of-two transform, the only
	// transforms it measured faster on (see stockham_amd64.go).
	for _, r := range []int{2, 4, 8} {
		k.Pass512(fb, fmt.Sprintf("skPass%dAVX512", r), r)
		k.Last512(fb, fmt.Sprintf("skLast%dAVX512", r), r)
	}
	for _, r := range []int{2, 3, 4, 5, 8} {
		k.Batch(fb, fmt.Sprintf("skBatch%dAVX2", r), r)
	}
	// Radix 16 at 256 bits only: the fft package never factors with radix 16
	// where the AVX-512 kernels run (see radix16TableAMD64).
	k.Pass(fb, "skPass16AVX2", 16)
	k.Last(fb, "skLast16AVX2", 16)
	// Radix 12 (Round 24), for composites, which never run the AVX-512 kernels.
	k.Pass(fb, "skPass12AVX2", 12)
	k.Last(fb, "skLast12AVX2", 12)
	// The final pass of the blocked schedule for large powers of two
	// (cascadeBlocked in the fft package): the radices that end a power of two.
	for _, r := range []int{2, 4, 8} {
		k.LastRun(fb, fmt.Sprintf("skLastRun%dAVX2", r), r)
		k.LastRun512(fb, fmt.Sprintf("skLastRun%dAVX512", r), r)
	}
	// The two passes it runs over the whole array chunk by chunk
	// (cascadePair): the radices that open a large power of two.
	for _, r := range []int{4, 8} {
		k.Strided(fb, fmt.Sprintf("skStrided%dAVX2", r), r)
		k.Strided512(fb, fmt.Sprintf("skStrided%dAVX512", r), r)
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
	// FMA3 (Round 41): a leaf-1 bit of its own, which a hypervisor can mask
	// apart from AVX2; go-asmgen v0.16.0's probe also checks that the OS
	// saves YMM state. It gates the FMA kernels below, which only plans built
	// with the fft package's Options.FMA run.
	fp.Add(amd64.FeatureProbe("supportsFMA", amd64.FMA))
	writeFile("cpu_amd64.s", fp.String())

	// The float32 pass kernels.
	f := emit.NewFile("amd64")
	for _, r := range []int{2, 3, 4, 5, 8} {
		k.Pass32(f, fmt.Sprintf("sk32Pass%dAVX2", r), r)
	}
	for _, r := range []int{2, 3, 4, 5, 8} {
		k.Last32(f, fmt.Sprintf("sk32Last%dAVX2", r), r)
	}
	writeFile("stockham32_amd64.s", f.String())

	// The split-layout pass kernels: every radix with a split body, every
	// pair of input and output layouts, both directions. Radix 3 and 5
	// (Round 36) come after the others so their code is unmoved.
	f = emit.NewFile("amd64")
	for _, r := range []int{4, 8, 3, 5} {
		forSplits(func(inS, outS, inverse bool) {
			k.Split(f, splitName("skSplit", r, inS, outS, inverse), r, inS, outS, inverse)
		})
	}
	writeFile("stockhamsplit_amd64.s", f.String())

	// The float32 real-FFT untangle and retangle, and the float32 batched
	// passes.
	f = emit.NewFile("amd64")
	k.Untangle32(f, "f32rUntangleAVX2")
	k.Retangle32(f, "f32rRetangleAVX2")
	writeFile("untangle32_amd64.s", f.String())
	f = emit.NewFile("amd64")
	for _, r := range []int{2, 3, 4, 5, 8} {
		k.Batch32(f, fmt.Sprintf("sk32Batch%dAVX2", r), r)
	}
	writeFile("batch32_amd64.s", f.String())

	// The radix-10, -15 and -20 kernels (Round 26).
	f = emit.NewFile("amd64")
	for _, r := range []int{10, 15, 20} {
		k.Pass(f, fmt.Sprintf("skPass%dAVX2", r), r)
		k.Last(f, fmt.Sprintf("skLast%dAVX2", r), r)
	}
	writeFile("comp2_amd64.s", f.String())

	// The 512-bit split-layout pass kernels (Round 28, Cascade Lake): radix 4
	// and 8, every pair of layouts, both directions.
	f = emit.NewFile("amd64")
	for _, r := range []int{4, 8} {
		forSplits(func(inS, outS, inverse bool) {
			k.Split512(f, splitName("intelSplit512R", r, inS, outS, inverse), r, inS, outS, inverse)
		})
	}
	writeFile("stockhamsplit512_amd64.s", f.String())

	// The float64 real-FFT untangle and retangle, at 256 and 512 bits; the
	// 512-bit kernels hand what is left to the 256-bit ones by a tail jump.
	f = emit.NewFile("amd64")
	k.Untangle(f, "smallUntangleAVX2", false)
	k.Untangle(f, "smallRetangleAVX2", true)
	k.Untangle512(f, "clUntangleAVX512", "smallUntangleAVX2", false)
	k.Untangle512(f, "clRetangleAVX512", "smallRetangleAVX2", true)
	writeFile("untangle_amd64.s", f.String())

	// Two- and three-pass transforms on the kernel's own stack frame (Round
	// 32, 35), and the table of the kernels' entry points they call.
	f = emit.NewFile("amd64")
	fft.TwoPass(f, "znTwoPassAVX2", znTwoPassMax)
	fft.ThreePass(f, "znThreePassAVX2", znThreePassMax)
	fft.AddrTable(f, "znKernelAddrs", znAddrs())
	writeFile("zntwopass_amd64.s", fft.Funcdata(f.String()))

	// Radix 9 (Round 35), the cs kernels.
	f = emit.NewFile("amd64")
	k.Pass(f, "csPass9AVX2", 9)
	k.Last(f, "csLast9AVX2", 9)
	writeFile("radix9_amd64.s", f.String())

	// Bluestein's pointwise products (Round 36, kernels.CMulTo).
	f = emit.NewFile("amd64")
	fft.CMulAVX2(f, "zmCMulAVX2")
	writeFile("zmcmul_amd64.s", f.String())

	genFMAFiles()
	genR32Files()
}

// genR32Files writes r32fma_amd64.s (Round 42): the radix-32 pass kernels
// of the FMA plans, which only they run. The split-layout pass at 256 and
// 512 bits, every pair of layouts, both directions; the final pass,
// interleaved, at 256 bits.
func genR32Files() {
	f := emit.NewFile("amd64")
	forSplits(func(inS, outS, inverse bool) {
		kf.SplitR32(f, splitName("skSplit", 32, inS, outS, inverse)+"FMA", false, inS, outS, inverse)
	})
	forSplits(func(inS, outS, inverse bool) {
		kf.SplitR32(f, splitName("intelSplit512R", 32, inS, outS, inverse)+"FMA", true, inS, outS, inverse)
	})
	kf.LastR32(f, "skLast32AVX2FMA")
	writeFile("r32fma_amd64.s", f.String())
}

// genFMAFiles writes the FMA kernels (Round 41): fma_amd64.s (interleaved:
// the AVX2, AVX-512, strided, batched, radix-9, -12, -16 and prime-factor
// pass kernels), fmasplit_amd64.s and fmasplit512_amd64.s (the split
// layouts). Every pass kernel that multiplies by twiddles is generated a
// second time with the products fused (fft.Kernels.FMA); the butterflies,
// the loops, the layouts and the blends are the default kernels' own, so
// each FMA kernel differs from its twin in the products alone, and its
// reference is the fft package's pass with cmulFMA in place of the complex
// product. The names are the default kernels' with "FMA" appended.
func genFMAFiles() {
	f := emit.NewFile("amd64")
	for _, r := range []int{2, 3, 4, 5, 8} {
		kf.Pass(f, fmt.Sprintf("skPass%dAVX2FMA", r), r)
	}
	for _, r := range []int{2, 4, 8} {
		kf.Pass512(f, fmt.Sprintf("skPass%dAVX512FMA", r), r)
	}
	for _, r := range []int{2, 3, 4, 5, 8} {
		kf.Batch(f, fmt.Sprintf("skBatch%dAVX2FMA", r), r)
	}
	kf.Pass(f, "skPass16AVX2FMA", 16)
	kf.Pass(f, "skPass12AVX2FMA", 12)
	for _, r := range []int{4, 8} {
		kf.Strided(f, fmt.Sprintf("skStrided%dAVX2FMA", r), r)
		kf.Strided512(f, fmt.Sprintf("skStrided%dAVX512FMA", r), r)
	}
	for _, r := range []int{10, 15, 20} {
		kf.Pass(f, fmt.Sprintf("skPass%dAVX2FMA", r), r)
	}
	kf.Pass(f, "csPass9AVX2FMA", 9)
	// The final passes whose butterflies have a constant product that the
	// FMA kernels fuse (radix 8: √2/2; 16 and 9: their internal twiddles).
	// The other radices' final passes are the default kernels.
	kf.Last(f, "skLast8AVX2FMA", 8)
	kf.Last512(f, "skLast8AVX512FMA", 8)
	kf.Last(f, "skLast16AVX2FMA", 16)
	kf.Last(f, "csLast9AVX2FMA", 9)
	kf.LastRun(f, "skLastRun8AVX2FMA", 8)
	kf.LastRun512(f, "skLastRun8AVX512FMA", 8)
	writeFile("fma_amd64.s", f.String())

	f = emit.NewFile("amd64")
	for _, r := range []int{4, 8, 3, 5} {
		forSplits(func(inS, outS, inverse bool) {
			kf.Split(f, splitName("skSplit", r, inS, outS, inverse)+"FMA", r, inS, outS, inverse)
		})
	}
	writeFile("fmasplit_amd64.s", f.String())

	f = emit.NewFile("amd64")
	for _, r := range []int{4, 8} {
		forSplits(func(inS, outS, inverse bool) {
			kf.Split512(f, splitName("intelSplit512R", r, inS, outS, inverse)+"FMA", r, inS, outS, inverse)
		})
	}
	writeFile("fmasplit512_amd64.s", f.String())
}

// znTwoPassMax and znThreePassMax are the largest transforms the two- and
// three-pass frames hold, in complex128 points; they must match the kernels
// package's ZnTwoPassMax and ZnThreePassMax.
const (
	znTwoPassMax   = 256
	znThreePassMax = 512
)

// znRadices are the radices with AVX2 pass and final-pass kernels.
var znRadices = []int{2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 20}

// znSplitModes names the split kernels' layouts by mode (1..4).
var znSplitModes = []string{1: "IS", 2: "SS", 3: "SI", 4: "II"}

// znAddrs is the layout of znKernelAddrs(t *[znAddrSlots]uintptr): the
// interleaved pass kernels of radix r at t[r], the final-pass kernels at
// t[21+r], the split kernels of radix r, mode m and direction d (0 forward)
// at t[42+(9d+r)·5+m], the 512-bit split kernels at 132 + their 256-bit
// twins' offset, and the AVX-512 final-pass kernels by radix from 222.
func znAddrs() []fft.Addr {
	var t []fft.Addr
	put := func(sym string, slot int) { t = append(t, fft.Addr{Sym: sym, Slot: slot}) }
	for _, r := range znRadices {
		pre := "sk"
		if r == 9 {
			pre = "cs" // Round 35's kernels
		}
		put(fmt.Sprintf("%sPass%dAVX2", pre, r), r)
		put(fmt.Sprintf("%sLast%dAVX2", pre, r), 21+r)
	}
	for d, dir := range []string{"Fwd", "Inv"} {
		for _, r := range []int{4, 8, 3, 5} {
			for m := 1; m <= 4; m++ {
				put(fmt.Sprintf("skSplit%d%s%s", r, znSplitModes[m], dir), 42+(9*d+r)*5+m)
			}
		}
	}
	// Round 35: the 512-bit split kernels (Cascade Lake's powers of two from
	// 256 points) and the AVX-512 final-pass kernels.
	for d, dir := range []string{"Fwd", "Inv"} {
		for _, r := range []int{4, 8} {
			for m := 1; m <= 4; m++ {
				put(fmt.Sprintf("intelSplit512R%d%s%s", r, znSplitModes[m], dir), 132+(9*d+r)*5+m)
			}
		}
	}
	for _, r := range []int{2, 4, 8} {
		put(fmt.Sprintf("skLast%dAVX512", r), 222+r)
	}
	// Round 41: the kernels of the FMA plans (genFMAFiles), each at 243 +
	// its default twin's slot, the same layout again. A final pass with no
	// product to fuse (every radix but 8, 9 and 16) has no twin: its slot
	// holds the default kernel, which the FMA plans run.
	const fmaAt = 243
	fused := map[int]bool{8: true, 9: true, 16: true}
	for _, r := range znRadices {
		pre := "sk"
		if r == 9 {
			pre = "cs"
		}
		put(fmt.Sprintf("%sPass%dAVX2FMA", pre, r), fmaAt+r)
		last := fmt.Sprintf("%sLast%dAVX2", pre, r)
		if fused[r] {
			last += "FMA"
		}
		put(last, fmaAt+21+r)
	}
	for _, r := range []int{2, 4, 8} {
		last := fmt.Sprintf("skLast%dAVX512", r)
		if fused[r] {
			last += "FMA"
		}
		put(last, fmaAt+222+r)
	}
	for d, dir := range []string{"Fwd", "Inv"} {
		for _, r := range []int{4, 8, 3, 5} {
			for m := 1; m <= 4; m++ {
				put(fmt.Sprintf("skSplit%d%s%sFMA", r, znSplitModes[m], dir), fmaAt+42+(9*d+r)*5+m)
			}
		}
		for _, r := range []int{4, 8} {
			for m := 1; m <= 4; m++ {
				put(fmt.Sprintf("intelSplit512R%d%s%sFMA", r, znSplitModes[m], dir), fmaAt+132+(9*d+r)*5+m)
			}
		}
	}
	return t
}

// forSplits calls emit for every split kernel of one radix, in the order of
// the committed files: forward then inverse, and the layouts IS, SS, SI, II.
func forSplits(emit func(inS, outS, inverse bool)) {
	for _, inverse := range []bool{false, true} {
		for _, io := range [][2]bool{{false, true}, {true, true}, {true, false}, {false, false}} {
			emit(io[0], io[1], inverse)
		}
	}
}

// splitName is a split kernel's symbol: prefix, the radix, the input and
// output layouts (I interleaved, S split) and the direction.
func splitName(prefix string, r int, inS, outS, inverse bool) string {
	io := map[bool]string{false: "I", true: "S"}
	dir := "Fwd"
	if inverse {
		dir = "Inv"
	}
	return fmt.Sprintf("%s%d%s%s%s", prefix, r, io[inS], io[outS], dir)
}

func writeFile(name, content string) {
	if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

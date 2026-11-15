//go:build ignore

// Command genfused256 writes internal/kernels/fused256_amd64.s: a complex128
// 256-point FFT as two fused 16x16 passes on AVX-512 (go-fft/fft#124).
//
// x[16a+b], X[k1+16*k2]:  step 1 runs, for each group of four columns b, a
// 16-point DFT across sixteen zmm registers (the a axis), multiplies by
// w256^(k1*b) from a table, and 4x4-transposes the result into a 4 KB frame
// (the only memory the data touches between the passes); step 2 reloads sixteen
// registers per block of four k1 and runs the 16-point DFT across them (the b
// axis), storing contiguous rows. The 16-point DFT is 4x4: radix-4, w16
// twiddles, radix-4, every value in a zmm register, registers renamed from a
// free list instead of moved.
//
// Run from the package root with `go generate` (see plan.go) or
// `go run internal/genfused256/main.go`.
package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

var (
	out    strings.Builder
	free   []int
	consts = map[string]float64{}
	inv    bool
)

func emit(f string, a ...any) { fmt.Fprintf(&out, "\t"+f+"\n", a...) }

func alloc() int {
	r := free[len(free)-1]
	free = free[:len(free)-1]
	return r
}
func release(r int)  { free = append(free, r) }
func z(r int) string { return fmt.Sprintf("Z%d", r) }

// correctly rounded cos/sin of 2*pi*m/16 via big arithmetic of the exact
// angle is overkill for 16ths; math.Cos of the octant-reduced angle is within
// 1 ulp, and the exact values for multiples of pi/4 are set explicitly.
func cs(m int) (c, s float64) {
	m &= 15
	switch m {
	case 0:
		return 1, 0
	case 4:
		return 0, 1
	case 8:
		return -1, 0
	case 12:
		return 0, -1
	}
	r := math.Sqrt2 / 2
	switch m {
	case 2:
		return r, r
	case 6:
		return -r, r
	case 10:
		return -r, -r
	case 14:
		return r, -r
	}
	ang := 2 * math.Pi * float64(m) / 16
	return math.Cos(ang), math.Sin(ang)
}

func cname(v float64) string {
	n := fmt.Sprintf("f256_%016x<>", math.Float64bits(v))
	consts[n] = v
	return n
}

// radix4 maps four registers to the four outputs of a 4-point DFT (forward or
// inverse), freeing the inputs.
func radix4(a0, a1, a2, a3 int) (o0, o1, o2, o3 int) {
	t0, t1, t2, t3 := alloc(), alloc(), alloc(), alloc()
	emit("VADDPD %s, %s, %s", z(a2), z(a0), z(t0))
	emit("VSUBPD %s, %s, %s", z(a2), z(a0), z(t1))
	emit("VADDPD %s, %s, %s", z(a3), z(a1), z(t2))
	emit("VSUBPD %s, %s, %s", z(a3), z(a1), z(t3))
	release(a0)
	release(a1)
	release(a2)
	release(a3)
	o0, o2 = alloc(), alloc()
	emit("VADDPD %s, %s, %s", z(t2), z(t0), z(o0))
	emit("VSUBPD %s, %s, %s", z(t2), z(t0), z(o2))
	release(t0)
	release(t2)
	sw1, sw3 := alloc(), alloc()
	emit("VPERMILPD $0x55, %s, %s", z(t3), z(sw1))
	emit("VPERMILPD $0x55, %s, %s", z(t3), z(sw3))
	release(t3)
	one := cname(1)
	o1, o3 = sw1, sw3
	// s1 -/+ i*s3, with sw = swap(s3) = (s3im, s3re):
	//   s1 - i*s3 = (s1re + s3im, s1im - s3re): even +, odd -  (fmsubadd)
	//   s1 + i*s3 = (s1re - s3im, s1im + s3re): even -, odd +  (fmaddsub)
	m1, m3 := "VFMSUBADD231PD", "VFMADDSUB231PD"
	if inv {
		m1, m3 = m3, m1
	}
	emit("%s.BCST %s(SB), %s, %s", m1, one, z(t1), z(o1))
	emit("%s.BCST %s(SB), %s, %s", m3, one, z(t1), z(o3))
	release(t1)
	return
}

// cmulConst multiplies register r by w16^m (forward: e^-2 pi i m/16) and
// returns the register holding the product.
func cmulConst(r, m int) int {
	m &= 15
	if m == 0 {
		return r
	}
	c, s := cs(m)
	if !inv {
		s = -s
	}
	if m == 8 {
		t := alloc()
		emit("VXORPD.BCST %s(SB), %s, %s", cname(math.Copysign(0, -1)), z(r), z(t))
		release(r)
		return t
	}
	t := alloc()
	emit("VPERMILPD $0x55, %s, %s", z(r), z(t))
	emit("VMULPD.BCST %s(SB), %s, %s", cname(s), z(t), z(t))
	emit("VFMADDSUB231PD.BCST %s(SB), %s, %s", cname(c), z(r), z(t))
	release(r)
	return t
}

// dft16 takes sixteen register numbers (natural order) and returns the
// sixteen outputs (natural order).
func dft16(in [16]int) (outv [16]int) {
	var u [4][4]int // u[k1][j2]
	for j2 := 0; j2 < 4; j2++ {
		u[0][j2], u[1][j2], u[2][j2], u[3][j2] = radix4(in[j2], in[4+j2], in[8+j2], in[12+j2])
	}
	for k1 := 1; k1 < 4; k1++ {
		for j2 := 1; j2 < 4; j2++ {
			u[k1][j2] = cmulConst(u[k1][j2], j2*k1)
		}
	}
	for k1 := 0; k1 < 4; k1++ {
		outv[k1], outv[k1+4], outv[k1+8], outv[k1+12] = radix4(u[k1][0], u[k1][1], u[k1][2], u[k1][3])
	}
	return
}

func transpose4(a0, a1, a2, a3 int) (o0, o1, o2, o3 int) {
	t0, t1, t2, t3 := alloc(), alloc(), alloc(), alloc()
	emit("VSHUFF64X2 $0x88, %s, %s, %s", z(a1), z(a0), z(t0))
	emit("VSHUFF64X2 $0xDD, %s, %s, %s", z(a1), z(a0), z(t1))
	emit("VSHUFF64X2 $0x88, %s, %s, %s", z(a3), z(a2), z(t2))
	emit("VSHUFF64X2 $0xDD, %s, %s, %s", z(a3), z(a2), z(t3))
	release(a0)
	release(a1)
	release(a2)
	release(a3)
	o0, o1, o2, o3 = alloc(), alloc(), alloc(), alloc()
	emit("VSHUFF64X2 $0x88, %s, %s, %s", z(t2), z(t0), z(o0))
	emit("VSHUFF64X2 $0x88, %s, %s, %s", z(t3), z(t1), z(o1))
	emit("VSHUFF64X2 $0xDD, %s, %s, %s", z(t2), z(t0), z(o2))
	emit("VSHUFF64X2 $0xDD, %s, %s, %s", z(t3), z(t1), z(o3))
	release(t0)
	release(t1)
	release(t2)
	release(t3)
	return
}

func fn(name string, inverse bool) {
	inv = inverse
	free = free[:0]
	for r := 31; r >= 0; r-- {
		free = append(free, r)
	}
	fmt.Fprintf(&out, "\n// func %s(dst, src, tw *complex128)\nTEXT ·%s(SB), $4160-24\n", name, name)
	emit("MOVQ dst+0(FP), DI")
	emit("MOVQ src+8(FP), SI")
	emit("MOVQ tw+16(FP), DX")
	emit("MOVQ SP, R8")
	emit("ADDQ $63, R8")
	emit("ANDQ $-64, R8")
	// step 1
	for g := 0; g < 4; g++ {
		var in [16]int
		for a := 0; a < 16; a++ {
			in[a] = alloc()
			emit("VMOVUPD %d(SI), %s", (16*a+4*g)*16, z(in[a]))
		}
		y := dft16(in)
		for k1 := 1; k1 < 16; k1++ {
			off := (k1*4 + g) * 128
			t := alloc()
			emit("VPERMILPD $0x55, %s, %s", z(y[k1]), z(t))
			emit("VMULPD %d(DX), %s, %s", off+64, z(t), z(t))
			emit("VFMADDSUB231PD %d(DX), %s, %s", off, z(y[k1]), z(t))
			release(y[k1])
			y[k1] = t
		}
		for h := 0; h < 4; h++ {
			o0, o1, o2, o3 := transpose4(y[4*h], y[4*h+1], y[4*h+2], y[4*h+3])
			for j, r := range []int{o0, o1, o2, o3} {
				emit("VMOVUPD %s, %d(R8)", z(r), (h*16+4*g+j)*64)
				release(r)
			}
		}
	}
	// step 2
	for h := 0; h < 4; h++ {
		var in [16]int
		for b := 0; b < 16; b++ {
			in[b] = alloc()
			emit("VMOVUPD %d(R8), %s", (h*16+b)*64, z(in[b]))
		}
		x := dft16(in)
		for k2 := 0; k2 < 16; k2++ {
			emit("VMOVUPD %s, %d(DI)", z(x[k2]), (16*k2+4*h)*16)
			release(x[k2])
		}
	}
	emit("VZEROUPPER")
	emit("RET")
}

func main() {
	fn("fused256Fwd", false)
	fn("fused256Inv", true)
	var hdr strings.Builder
	hdr.WriteString("// Code generated by internal/genfused256/main.go; DO NOT EDIT.\n\n//go:build amd64\n\n#include \"textflag.h\"\n\n")
	names := make([]string, 0, len(consts))
	for n := range consts {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Fprintf(&hdr, "DATA %s+0(SB)/8, $%#x\nGLOBL %s(SB), RODATA|NOPTR, $8\n", n, math.Float64bits(consts[n]), n)
	}
	path := "internal/kernels/fused256_amd64.s"
	if len(os.Args) > 1 {
		path = os.Args[1] // the staleness test writes elsewhere
	}
	if err := os.WriteFile(path, []byte(hdr.String()+out.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

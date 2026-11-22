//go:build ignore

// The float32 amd64 kernels of the fused DCT-IV of length 64 (go-avkit/decode#77).
//
// The same program as the float64 pair kernels (cbuild), run in single precision:
// one register holds one complex value of S independent transforms, lanes
// (re, im) per stream. S = 2 uses 128-bit registers (a pair), S = 4 the 256-bit
// ones (a quad: four transforms for the instruction count of a pair), so a quad
// costs about half as much per transform. Every twiddle is a constant row of
// eight float32 (the 2-periodic pattern of the float64 rows repeated), the product
// a*w is VPERMILPS $0xB1, VMULPS, VFMADD231PS. There is no conversion: the loads
// gather (in[2m], in[n-1-2m]) of each stream with VMOVSS and VINSERTPS, the stores
// write the two outputs of a stream with VMOVQ and VMOVHPS.
package main

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
)

// asmFunc32 returns the assembly of the float32 kernel of s streams (2 or 4).
func (p *cprog) asmFunc32(name string, s int, scaled bool) string {
	var b bytes.Buffer
	pfx, mov := "X", "VMOVUPS"
	if s == 4 {
		pfx = "Y"
	}
	ra := &yalloc{b: &b, reg: make([]int, p.nv), slot: make([]int, p.nv), uses: make([][]int, p.nv), mov: mov, pfx: pfx}
	for i := range ra.reg {
		ra.reg[i], ra.slot[i] = -1, -1
	}
	for i := range ra.holder {
		ra.holder[i] = -1
	}
	for i, c := range p.code {
		switch c.op {
		case cAdd, cSub, cSt2:
			ra.uses[c.a] = append(ra.uses[c.a], i)
			ra.uses[c.b] = append(ra.uses[c.b], i)
		case cMul, cNegI, cMulf:
			ra.uses[c.a] = append(ra.uses[c.a], i)
		}
	}
	dst := []string{"DI", "SI", "R8", "R9"}[:s]
	src := []string{"AX", "BX", "R10", "R11"}[:s]
	argSize := 8 * (2*s + 1)
	for i := range dst {
		fmt.Fprintf(&b, "\tMOVQ dst%d+%d(FP), %s\n", i, 8*i, dst[i])
	}
	for i := range src {
		fmt.Fprintf(&b, "\tMOVQ src%d+%d(FP), %s\n", i, 8*(s+i), src[i])
	}
	fmt.Fprintf(&b, "\tMOVQ w+%d(FP), DX\n", 8*2*s)
	if scaled {
		name += "S"
		fmt.Fprintf(&b, "\tVBROADCASTSS f+%d(FP), %s14\n", 8*(2*s+1), pfx)
		argSize += 4
	}
	row := func(k int) string { return fmt.Sprintf("%d(DX)", 32*k) }
	tmp := fmt.Sprintf("%s%d", pfx, yTmp)
	for i, c := range p.code {
		ra.pos = i
		switch c.op {
		case cLd:
			d := ra.alloc(c.d)
			ia, ib := 2*c.i, n-1-2*c.i
			// stream t fills lanes 2t (in[ia]) and 2t+1 (in[ib]); streams 0, 1 into the
			// low 128 bits of d, streams 2, 3 into the low 128 bits of the scratch.
			ld := func(x string, regs []string) {
				fmt.Fprintf(&b, "\tVMOVSS %d(%s), X%s\n", 4*ia, regs[0], x)
				fmt.Fprintf(&b, "\tVINSERTPS $0x10, %d(%s), X%s, X%s\n", 4*ib, regs[0], x, x)
				fmt.Fprintf(&b, "\tVINSERTPS $0x20, %d(%s), X%s, X%s\n", 4*ia, regs[1], x, x)
				fmt.Fprintf(&b, "\tVINSERTPS $0x30, %d(%s), X%s, X%s\n", 4*ib, regs[1], x, x)
			}
			ld(strconv.Itoa(d), src[:2])
			if s == 4 {
				ld(strconv.Itoa(yTmp), src[2:])
				fmt.Fprintf(&b, "\tVINSERTF128 $1, X%d, Y%d, Y%d\n", yTmp, d, d)
			}
		case cAdd, cSub:
			a, bb := c.a, c.b
			if c.op == cAdd && ra.reg[a] < 0 && ra.reg[bb] >= 0 {
				a, bb = bb, a
			}
			ya := ra.use(a, bb)
			var bt string
			if r := ra.reg[bb]; r >= 0 {
				bt = fmt.Sprintf("%s%d", pfx, r)
			} else {
				bt = fmt.Sprintf("%d(SP)", 32*ra.slot[bb])
			}
			d := ra.take(c.d, a, a, bb)
			mn := "VADDPS"
			if c.op == cSub {
				mn = "VSUBPS"
			}
			fmt.Fprintf(&b, "\t%s %s, %s%d, %s%d\n", mn, bt, pfx, ya, pfx, d)
		case cMul:
			ya := ra.use(c.a)
			fmt.Fprintf(&b, "\tVPERMILPS $0xB1, %s%d, %s\n", pfx, ya, tmp)
			d := ra.take(c.d, c.a, c.a)
			fmt.Fprintf(&b, "\tVMULPS %s, %s%d, %s%d\n\tVFMADD231PS %s, %s, %s%d\n", row(c.k1), pfx, ya, pfx, d, row(c.k2), tmp, pfx, d)
		case cNegI:
			ya := ra.use(c.a)
			d := ra.take(c.d, c.a, c.a)
			fmt.Fprintf(&b, "\tVPERMILPS $0xB1, %s%d, %s%d\n\tVXORPS %s, %s%d, %s%d\n", pfx, ya, pfx, d, row(c.k1), pfx, d, pfx, d)
		case cMulf:
			ya := ra.use(c.a)
			d := ra.take(c.d, c.a, c.a)
			fmt.Fprintf(&b, "\tVMULPS %s14, %s%d, %s%d\n", pfx, pfx, ya, pfx, d)
		case cSt2:
			ya := ra.use(c.a, c.b)
			yb := ra.use(c.b, c.a)
			store := func(idx int) {
				fmt.Fprintf(&b, "\tVMOVQ X%d, %d(%s)\n\tVMOVHPS X%d, %d(%s)\n", yTmp, 4*idx, dst[0], yTmp, 4*idx, dst[1])
				if s == 4 {
					fmt.Fprintf(&b, "\tVEXTRACTF128 $1, Y%d, X%d\n\tVMOVQ X%d, %d(%s)\n\tVMOVHPS X%d, %d(%s)\n", yTmp, yTmp, yTmp, 4*idx, dst[2], yTmp, 4*idx, dst[3])
				}
			}
			// even lanes from the first register, odd lanes from the second
			fmt.Fprintf(&b, "\tVBLENDPS $0xAA, %s%d, %s%d, %s\n", pfx, yb, pfx, ya, tmp)
			store(c.i)
			fmt.Fprintf(&b, "\tVBLENDPS $0xAA, %s%d, %s%d, %s\n", pfx, ya, pfx, yb, tmp)
			store(c.j)
		}
		switch c.op {
		case cAdd, cSub, cSt2:
			ra.release(c.a)
			ra.release(c.b)
		case cMul, cNegI, cMulf:
			ra.release(c.a)
		}
	}
	b.WriteString("\tVZEROUPPER\n\tRET\n\n")
	return fmt.Sprintf("TEXT ·%s(SB), 0, $%d-%d\n", name, 32*ra.nslots+32, argSize) + b.String()
}

// amd64Gen32 writes the float32 kernels and their constant table.
func amd64Gen32() {
	var sSrc bytes.Buffer
	for _, v := range []struct {
		name string
		s    int
	}{{"dct4n64Y32Pair", 2}, {"dct4n64Y32Quad", 4}} {
		for _, scaled := range []bool{false, true} {
			sSrc.WriteString(cbuild(scaled).asmFunc32(v.name, v.s, scaled))
		}
	}
	var vt bytes.Buffer
	vt.WriteString("// Code generated by internal/gendct4; DO NOT EDIT.\n\npackage fft\n\nimport \"math\"\n\n")
	fmt.Fprintf(&vt, "// dct4n64Y32Tab holds the %d distinct 256-bit constant rows of float32 the amd64 float32 kernels read.\ntype dct4n64Y32Tab [%d][8]float32\n\n", len(yRows), len(yRows))
	vt.WriteString("var dct4n64Y32 = dct4n64Y32Tab{\n")
	for _, r := range yRows {
		vt.WriteString("{")
		for i := 0; i < 8; i++ {
			x := float32(r[i%2])
			s := strconv.FormatFloat(float64(x), 'g', -1, 32)
			if x == 0 && math.Signbit(float64(x)) {
				s = "math.Float32frombits(0x80000000)"
			} else {
				s = "float32(" + s + ")"
			}
			if i > 0 {
				vt.WriteString(", ")
			}
			vt.WriteString(s)
		}
		vt.WriteString("},\n")
	}
	vt.WriteString("}\n")
	write("dct4_64_gen32_amd64.go", vt.Bytes(), true)
	write("dct4_64_f32_amd64.s", []byte("// Code generated by internal/gendct4; DO NOT EDIT.\n\n#include \"textflag.h\"\n\n"+sSrc.String()), false)
}

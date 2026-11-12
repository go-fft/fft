package fft

import (
	"fmt"
	"runtime"
	"strconv"
)

// r2rCase is one DCT, DST or MDCT call. These transforms take no Options, so
// they cannot request Exact: the digests pin the DEFAULT output as released
// at v0.29.1 (the DCT and DST bits equal v0.29.0's on arm64; the MDCT of n >=
// 64 changed between v0.29.0 and v0.29.1, when #132 fused it), until a design
// says how an exact mode reaches them (#121 step 3).
type r2rCase struct {
	name string
	run  func() any
}

func r2rCases() []r2rCase {
	var cs []r2rCase
	key := ""
	add := func(name string, f func() any) { cs = append(cs, r2rCase{name + "/" + key, f}) }
	lengths := []int{2, 3, 8, 17, 100, 257, 1000, 1009, 4096}
	shapes := [][]int{{8, 9}, {16, 17}, {5, 6, 7}, {64, 64}}
	mdct := []int{2, 4, 6, 8, 12, 30, 64, 100, 256, 1000, 2048}
	switch {
	case runtime.GOARCH == "s390x" || runtime.GOARCH == "ppc64le" || runtime.GOARCH == "loong64" || runtime.GOARCH == "riscv64" || raceEnabled:
		lengths, shapes, mdct = lengths[:6], shapes[:3], mdct[:8]
	}
	norms := []Norm{NormBackward, NormOrtho, NormForward}
	for _, n := range lengths {
		x := fmaReal(n, uint64(n))
		x32 := make([]float32, n)
		for i, v := range x {
			x32[i] = float32(v)
		}
		for typ := 1; typ <= 4; typ++ {
			if typ == 1 && n < 2 {
				continue
			}
			for _, norm := range norms {
				key = "typ=" + strconv.Itoa(typ) + "/" + norm.String() + "/n=" + strconv.Itoa(n)
				add("DCT", func() any { return DCT(x, typ, norm) })
				add("IDCT", func() any { return IDCT(x, typ, norm) })
				add("DST", func() any { return DST(x, typ, norm) })
				add("IDST", func() any { return IDST(x, typ, norm) })
				add("DCT32", func() any { return DCT32(x32, typ, norm) })
				add("IDST32", func() any { return IDST32(x32, typ, norm) })
				add("DCTPlan", func() any { return NewDCTPlan(n, typ).DCT(make([]float64, n), x, norm) })
				add("DSTPlan.IDST", func() any { return NewDSTPlan(n, typ).IDST(make([]float64, n), x, norm) })
			}
		}
	}
	for _, sh := range shapes {
		size := shapeProduct(sh...)
		x := fmaReal(size, uint64(size))
		x32 := make([]float32, size)
		for i, v := range x {
			x32[i] = float32(v)
		}
		for typ := 1; typ <= 4; typ++ {
			for _, norm := range []Norm{NormBackward, NormOrtho} {
				key = fmt.Sprint("typ=", typ, "/", norm.String(), "/shape=", sh)
				add("DCTN", func() any { return DCTN(x, sh, typ, norm) })
				add("IDCTN", func() any { return IDCTN(x, sh, typ, norm) })
				add("DSTN", func() any { return DSTN(x, sh, typ, norm) })
				add("IDSTN", func() any { return IDSTN(x, sh, typ, norm) })
				add("DCTN32", func() any { return DCTN32(x32, sh, typ, norm) })
			}
		}
	}
	for _, n := range mdct {
		x := fmaReal(n, uint64(n)+1)
		c := fmaReal(n/2, uint64(n)+2)
		x32 := make([]float32, n)
		c32 := make([]float32, n/2)
		for i, v := range x {
			x32[i] = float32(v)
		}
		for i, v := range c {
			c32[i] = float32(v)
		}
		for _, scale := range []float64{1, 0.5} {
			key = fmt.Sprint("scale=", scale, "/n=", n)
			add("MDCT", func() any { return NewMDCTPlan(n).MDCT(make([]float64, n/2), x, scale) })
			add("IMDCT", func() any { return NewMDCTPlan(n).IMDCT(make([]float64, n), c, scale) })
			add("MDCT32", func() any { return NewMDCTPlan32(n).MDCT(make([]float32, n/2), x32, float32(scale)) })
			add("IMDCT32", func() any { return NewMDCTPlan32(n).IMDCT(make([]float32, n), c32, float32(scale)) })
		}
	}
	return cs
}

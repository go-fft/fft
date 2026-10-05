package fft

// The N-dimensional DCT and DST, as scipy.fft.dctn/idctn/dstn/idstn with
// every axis transformed: the 1-D transform of the same type and norm along
// each axis of a row-major (C-order) array in turn. Normalisation composes
// per axis, as in scipy, so IDCTN(DCTN(x)) == x for every norm.

// DCTN returns the N-dimensional DCT of type typ of x, a row-major array of
// the given shape, as scipy.fft.dctn(x, type=typ, norm=norm). x is not
// modified. It panics if the shape does not match len(x), if an axis is
// below the type's minimum length, or on an unknown type or norm.
func DCTN(x []float64, shape []int, typ int, norm Norm) []float64 {
	return r2rN(x, shape, true, typ, norm, false)
}

// IDCTN inverts DCTN, as scipy.fft.idctn(x, type=typ, norm=norm).
func IDCTN(x []float64, shape []int, typ int, norm Norm) []float64 {
	return r2rN(x, shape, true, typ, norm, true)
}

// DSTN returns the N-dimensional DST of type typ of x, a row-major array of
// the given shape, as scipy.fft.dstn(x, type=typ, norm=norm). The same rules
// as DCTN apply.
func DSTN(x []float64, shape []int, typ int, norm Norm) []float64 {
	return r2rN(x, shape, false, typ, norm, false)
}

// IDSTN inverts DSTN, as scipy.fft.idstn(x, type=typ, norm=norm).
func IDSTN(x []float64, shape []int, typ int, norm Norm) []float64 {
	return r2rN(x, shape, false, typ, norm, true)
}

// r2rN validates everything first, then runs the 1-D transform along every
// axis of a copy of x, one line at a time through a gather buffer.
func r2rN(x []float64, shape []int, cosine bool, typ int, norm Norm, inverse bool) []float64 {
	checkR2R(cosine, typ, 2, norm) // the type and the norm, whatever the shape
	total := validateShape(shape, len(x))
	for _, s := range shape {
		checkR2R(cosine, typ, s, norm)
	}
	out := make([]float64, total)
	copy(out, x)
	stride := make([]int, len(shape))
	st := 1
	for a := len(shape) - 1; a >= 0; a-- {
		stride[a] = st
		st *= shape[a]
	}
	for ax, n := range shape {
		var run func(dst, src []float64, norm Norm) []float64
		switch {
		case cosine && inverse:
			run = cachedDCTPlan(n, typ).IDCT
		case cosine:
			run = cachedDCTPlan(n, typ).DCT
		case inverse:
			run = cachedDSTPlan(n, typ).IDST
		default:
			run = cachedDSTPlan(n, typ).DST
		}
		line := make([]float64, n)
		sa := stride[ax]
		for c := 0; c < total/n; c++ {
			base := lineBase(c, shape, stride, ax)
			for i := range line {
				line[i] = out[base+i*sa]
			}
			run(line, line, norm)
			for i, v := range line {
				out[base+i*sa] = v
			}
		}
	}
	return out
}

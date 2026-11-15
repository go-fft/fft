package fft

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
)

// digestAny is the SHA-256 of a result's bits (float bit patterns, NaNs
// canonicalized), as hex.
func digestAny(v any) string {
	h := sha256.New()
	var b [8]byte
	f64 := func(x float64) {
		if math.IsNaN(x) {
			x = math.NaN()
		}
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(x))
		h.Write(b[:])
	}
	f32 := func(x float32) {
		if x != x {
			x = float32(math.NaN())
		}
		binary.LittleEndian.PutUint32(b[:4], math.Float32bits(x))
		h.Write(b[:4])
	}
	switch x := v.(type) {
	case []complex128:
		for _, c := range x {
			f64(real(c))
			f64(imag(c))
		}
	case []float64:
		for _, c := range x {
			f64(c)
		}
	case []complex64:
		for _, c := range x {
			f32(real(c))
			f32(imag(c))
		}
	case []float32:
		for _, c := range x {
			f32(c)
		}
	default:
		panic(fmt.Sprintf("digestAny: %T", v))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func sameAny(a, b any) bool { return digestAny(a) == digestAny(b) }

// digestAnyZ is digestAny with every negative zero read as +0: the default
// and NoFMA scaling passes use the go-simd/floats kernel, whose zeros keep their
// sign where Exact's complex multiply turns some into +0 (scaleComplexA).
func digestAnyZ(v any) string {
	z := func(x float64) float64 {
		if x == 0 {
			return 0
		}
		return x
	}
	z32 := func(x float32) float32 {
		if x == 0 {
			return 0
		}
		return x
	}
	switch x := v.(type) {
	case []complex128:
		y := make([]complex128, len(x))
		for i, c := range x {
			y[i] = complex(z(real(c)), z(imag(c)))
		}
		return digestAny(y)
	case []float64:
		y := make([]float64, len(x))
		for i, c := range x {
			y[i] = z(c)
		}
		return digestAny(y)
	case []complex64:
		y := make([]complex64, len(x))
		for i, c := range x {
			y[i] = complex(z32(real(c)), z32(imag(c)))
		}
		return digestAny(y)
	case []float32:
		y := make([]float32, len(x))
		for i, c := range x {
			y[i] = z32(c)
		}
		return digestAny(y)
	}
	panic(fmt.Sprintf("digestAnyZ: %T", v))
}

func sameAnyZ(a, b any) bool { return digestAnyZ(a) == digestAnyZ(b) }

// zeroSignCase names the only two of the exact cases whose default (and NoFMA)
// result differs from Exact's, and only in the sign of a zero: the zero-padded
// one-point real transforms scaled by ortho or forward, whose scaling pass is
// the go-simd/floats kernel (scaleComplexA). Every other case is compared
// strictly.
func zeroSignCase(name string) bool {
	return name == "RFFTWith/pad/ortho/n=1" || name == "RFFTWith/pad/forward/n=1"
}

// digestCase is digestAny, reading negative zeros as +0 for the zeroSignCases.
func digestCase(name string, v any) string {
	if zeroSignCase(name) {
		return digestAnyZ(v)
	}
	return digestAny(v)
}

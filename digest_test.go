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

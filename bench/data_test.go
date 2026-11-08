package bench

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"strings"
)

// The inputs are built twice, once here and once by the Python expression that
// goes with them, from the same two formulas: frac(i*k) - 0.5 for a real part
// and for an imaginary part, k the golden ratio's fractional part and the
// plastic number's. float64(float64(i)*k) forbids a fused multiply-add, so the
// bytes are the same on arm64 and amd64. TestDataParity compares the SHA-256 of
// what Go builds with the SHA-256 NumPy gets from the expression.

const (
	kRe = 0.6180339887498949
	kIm = 0.7548776662466927
)

func frac(i int, k float64) float64 { return math.Mod(float64(float64(i)*k), 1) }

func reVec(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = frac(i, kRe) - 0.5
	}
	return x
}

func imVec(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = frac(i, kIm) - 0.5
	}
	return x
}

func pyRe(n int) string {
	return fmt.Sprintf("(np.mod(np.arange(%d, dtype=np.float64) * %v, 1.0) - 0.5)", n, kRe)
}

func pyIm(n int) string {
	return fmt.Sprintf("(np.mod(np.arange(%d, dtype=np.float64) * %v, 1.0) - 0.5)", n, kIm)
}

func pyStack(re, im string) string {
	return "np.ascontiguousarray(np.stack([" + re + ", " + im + "], axis=-1)).view(np.complex128).ravel()"
}

// rs appends a reshape to a Python expression.
func rs(expr string, shape ...int) string {
	s := make([]string, len(shape))
	for i, v := range shape {
		s[i] = fmt.Sprint(v)
	}
	return expr + ".reshape(" + strings.Join(s, ", ") + ")"
}

func prod(shape []int) int {
	p := 1
	for _, v := range shape {
		p *= v
	}
	return p
}

// A real signal of n points, float64 and float32.
func r64(n int, shape ...int) (string, []float64) {
	e := pyRe(n)
	if len(shape) > 0 {
		e = rs(e, shape...)
	}
	return e, reVec(n)
}

func r32(n int, shape ...int) (string, []float32) {
	e, x := r64(n, shape...)
	y := make([]float32, n)
	for i, v := range x {
		y[i] = float32(v)
	}
	return e + ".astype(np.float32)", y
}

// A complex signal of n points.
func c128(n int, shape ...int) (string, []complex128) {
	e := pyStack(pyRe(n), pyIm(n))
	if len(shape) > 0 {
		e = rs(e, shape...)
	}
	re, im := reVec(n), imVec(n)
	c := make([]complex128, n)
	for i := range c {
		c[i] = complex(re[i], im[i])
	}
	return e, c
}

func c64(n int, shape ...int) (string, []complex64) {
	e, x := c128(n, shape...)
	y := make([]complex64, n)
	for i, v := range x {
		y[i] = complex(float32(real(v)), float32(imag(v)))
	}
	return e + ".astype(np.complex64)", y
}

// A half spectrum of m bins whose first and last bins are real (what a real
// signal's spectrum has), for the real inverses and the Hermitian transforms.
func h128(m int) (string, []complex128) {
	mask := fmt.Sprintf("np.minimum(np.arange(%d), 1) * np.minimum(%d - np.arange(%d), 1)", m, m-1, m)
	e := pyStack(pyRe(m), pyIm(m)+" * "+mask)
	re, im := reVec(m), imVec(m)
	c := make([]complex128, m)
	for i := range c {
		w := float64(min(i, 1)) * float64(min(m-1-i, 1))
		c[i] = complex(re[i], im[i]*w)
	}
	return e, c
}

func h64(m int) (string, []complex64) {
	e, x := h128(m)
	y := make([]complex64, m)
	for i, v := range x {
		y[i] = complex(float32(real(v)), float32(imag(v)))
	}
	return e + ".astype(np.complex64)", y
}

// raw is the bytes NumPy hashes for the same array: little-endian, C order.
func raw(v any) []byte {
	var b []byte
	put32 := func(f float32) { b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f)) }
	put64 := func(f float64) { b = binary.LittleEndian.AppendUint64(b, math.Float64bits(f)) }
	switch t := v.(type) {
	case []float64:
		for _, f := range t {
			put64(f)
		}
	case []float32:
		for _, f := range t {
			put32(f)
		}
	case []complex128:
		for _, c := range t {
			put64(real(c))
			put64(imag(c))
		}
	case []complex64:
		for _, c := range t {
			put32(real(c))
			put32(imag(c))
		}
	default:
		panic(fmt.Sprintf("raw: %T", v))
	}
	return b
}

// flat is a result as the floats NumPy's flattened result holds: complex values
// interleaved, frames row after row.
func flat(v any) []float64 {
	var o []float64
	switch t := v.(type) {
	case []float64:
		o = t
	case []float32:
		for _, f := range t {
			o = append(o, float64(f))
		}
	case []complex128:
		for _, c := range t {
			o = append(o, real(c), imag(c))
		}
	case []complex64:
		for _, c := range t {
			o = append(o, float64(real(c)), float64(imag(c)))
		}
	case [][]float64:
		for _, r := range t {
			o = append(o, r...)
		}
	case int:
		o = []float64{float64(t)}
	default:
		panic(fmt.Sprintf("flat: %T", v))
	}
	return o
}

// spec is one row: what Go runs, what Python runs, what both are given.
type spec struct {
	name  string
	fn    string // the public function, in refbench's coverage form
	setup string // a Python list expression of the arrays both sides get
	call  string // a Python callable applied to them
	tol   float64
	ins   []any
	run   func(iters int)
	res   func() []float64
}

// mk builds a spec from a Go body returning its result. tol < 0: no value
// parity (the rows whose rule differs on purpose; see docs/perf.md).
func mk[O any](name, fn, setup, call string, tol float64, ins []any, f func() O) spec {
	return spec{
		name: name, fn: fn, setup: setup, call: call, tol: tol, ins: ins,
		run: func(iters int) {
			var s O
			for ; iters > 0; iters-- {
				s = f()
			}
			runtime.KeepAlive(s)
		},
		res: func() []float64 { return flat(f()) },
	}
}

// py wraps a lambda in a scope that names the scipy.fft module F, or
// scipy.signal as S.
func py(body string) string {
	return "(lambda F: " + body + ")(__import__('scipy.fft', fromlist=['_']))"
}

func pysig(body string) string {
	return "(lambda S: " + body + ")(__import__('scipy.signal', fromlist=['_']))"
}

func list(exprs ...string) string { return "[" + strings.Join(exprs, ", ") + "]" }

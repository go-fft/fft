package fft

import (
	"math"
	"strconv"
)

// Norm selects how a transform pair is scaled, as numpy.fft's and
// scipy.fft's norm argument does. The zero value is NormBackward, numpy's
// default and what FFT/IFFT do: the forward transform is unscaled and the
// inverse divides by n.
type Norm int

const (
	// NormBackward leaves the forward transform unscaled and scales the
	// inverse by 1/n (numpy norm="backward", the default).
	NormBackward Norm = iota
	// NormOrtho scales both directions by 1/sqrt(n), which makes the
	// transform unitary (numpy norm="ortho").
	NormOrtho
	// NormForward scales the forward transform by 1/n and leaves the inverse
	// unscaled (numpy norm="forward").
	NormForward
)

// String returns numpy's name for the mode: "backward", "ortho" or "forward".
func (m Norm) String() string {
	switch m {
	case NormBackward:
		return "backward"
	case NormOrtho:
		return "ortho"
	case NormForward:
		return "forward"
	}
	return "Norm(" + strconv.Itoa(int(m)) + ")"
}

// scale returns the factor a transform of n points applies under mode m, in
// the forward direction or, if inverse, the inverse one. It panics on a mode
// that is not one of the three constants, and returns 1 for n <= 0.
func (m Norm) scale(n int, inverse bool) float64 {
	if m < NormBackward || m > NormForward {
		panic("fft: unknown Norm " + m.String())
	}
	if n <= 0 {
		return 1
	}
	switch {
	case m == NormOrtho:
		return 1 / math.Sqrt(float64(n))
	case (m == NormBackward) == inverse:
		return 1 / float64(n)
	}
	return 1
}

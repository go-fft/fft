//go:build !arm64

package fft

// armrealRotateND is off here: the rotating passes were measured on arm64
// only.
var armrealRotateND = false

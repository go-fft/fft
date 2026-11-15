//go:build !amd64

package fft

// dct4PairAsmSwitch: there is no amd64 kernel switch off amd64.
func dct4PairAsmSwitch(bool) bool { return false }

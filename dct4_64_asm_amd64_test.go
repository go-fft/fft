package fft

// dct4PairAsmSwitch sets whether the pair forms run the AVX2 kernels and returns
// the old setting.
func dct4PairAsmSwitch(on bool) bool {
	old := dct4PairAsm
	dct4PairAsm = on
	return old
}

package kernels

// twiddle32At reads twiddle j of point i from an amd64 StockhamTwiddles32
// table: groups of g points (twGroups), each storing r-1 runs of g entries.
func twiddle32At(t []complex64, r, ido, i, j int) complex64 {
	at, i0 := 0, 0
	for _, g := range twGroups(ido) {
		if i < i0+g {
			return t[at+(j-1)*g+i-i0]
		}
		at += (r - 1) * g
		i0 += g
	}
	panic("point out of range")
}

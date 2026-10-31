package main

import "testing"

func TestClass(t *testing.T) {
	for _, c := range []struct{ line, want string }{
		{"VMOVUPD (AX), Y0", "load"},
		{"VMOVUPD Y0, (BX)(DX*1)", "store"},
		{"VMOVUPD Y1, Y2", "move"},
		{"VADDPD Y1, Y2, Y3", "add"},
		{"VFMADD231PD 0(R14), Y1, Y2", "fma"},
		{"VPERMILPD $5, Y1, Y2", "shuffle"},
		{"VINSERTF128 $1, 16(AX), Y0, Y0", "insert"},
		{"ADDQ $32, AX", "scalar"},
	} {
		m := c.line[:indexSpace(c.line)]
		if got := class(m, c.line); got != c.want {
			t.Errorf("%q: %s, want %s", c.line, got, c.want)
		}
	}
}

func indexSpace(s string) int {
	for i, r := range s {
		if r == ' ' {
			return i
		}
	}
	return len(s)
}

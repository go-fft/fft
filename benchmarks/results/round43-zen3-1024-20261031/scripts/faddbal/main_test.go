package main

import (
	"strings"
	"testing"
)

func TestRewrite(t *testing.T) {
	src := "TEXT ·k(SB), NOSPLIT, $0-8\n\tVADDPD Y2, Y1, Y1\n\tVSUBPD Y2, Y1, Y2\n\tVADDPD Y2, Y1, Y3\n\tVSUBPD Y2, Y1, Y3\n\tVADDPD 0(SP), Y1, Y3\nTEXT ·other(SB), NOSPLIT, $0-8\n\tVADDPD Y2, Y1, Y1\n"
	got, n := rewrite(src, map[string]bool{"k": true}, 1, 0, "O")
	if n != 4 {
		t.Fatalf("moved %d, want 4", n)
	}
	for _, want := range []string{
		"\tVFMADD132PD O, Y2, Y1",
		"\tVFNMADD132PD O, Y1, Y2",
		"\tVMOVAPD Y1, Y3\n\tVFMADD132PD O, Y2, Y3",
		"\tVMOVAPD Y1, Y3\n\tVFMSUB132PD O, Y2, Y3",
		"\tVADDPD 0(SP), Y1, Y3", // memory operand: untouched
		"TEXT ·other(SB), NOSPLIT, $0-8\n\tVADDPD Y2, Y1, Y1",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in\n%s", want, got)
		}
	}
}

#include "textflag.h"

// func cpuidSignature() uint32
// CPUID leaf 1, EAX: the stepping (bits 3:0), model (7:4), family (11:8) and
// extended model (19:16) and family (27:20). Hand written because go-asmgen has
// FamilyProbe and VendorProbe but no model probe (cpu_amd64.s is its output).
TEXT ·cpuidSignature(SB), NOSPLIT, $0-4
	MOVL $1, AX
	XORL CX, CX
	CPUID
	MOVL AX, ret+0(FP)
	RET

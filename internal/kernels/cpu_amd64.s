#include "textflag.h"

// Both the processor and the OS must support saving XMM/YMM state.
TEXT ·supportsAVX2(SB), NOSPLIT, $0-1
	MOVB $0, ret+0(FP)
	XORL AX, AX
	CPUID
	CMPL AX, $7
	JL done
	MOVL $1, AX
	CPUID
	ANDL $0x18000000, CX // AVX and OSXSAVE
	CMPL CX, $0x18000000
	JNE done
	XORL CX, CX
	XGETBV
	ANDL $6, AX // XMM and YMM state enabled in XCR0
	CMPL AX, $6
	JNE done
	MOVL $7, AX
	XORL CX, CX
	CPUID
	TESTL $0x20, BX // AVX2
	JZ done
	MOVB $1, ret+0(FP)
done:
	RET

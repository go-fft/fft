//go:build !amd64

package fft

func mdctKernelWindowDefault() (foldMin, foldMax, unfoldMin, unfoldMax int) {
	return 512, 4096, 256, 4096
}

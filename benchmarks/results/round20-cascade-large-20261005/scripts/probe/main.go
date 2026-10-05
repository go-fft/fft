//go:build linux && amd64

// probe characterises the memory system and the AVX frequency behaviour of
// the host it runs on (go-fft Round 20).
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"syscall"
	"time"
	"unsafe"
)

func readZ(p *byte, nbytes, reps int) float64
func readY(p *byte, nbytes, reps int) float64
func writeZ(p *byte, nbytes, reps int)
func copyZ(dst, src *byte, nbytes, reps int)
func copyY(dst, src *byte, nbytes, reps int)
func chain(n int) int
func chainZ(n int) int
func chainY(n int) int

// alloc maps nbytes (rounded to 2 MiB) aligned to 2 MiB, advised huge or not.
func alloc(nbytes int, huge bool) []byte {
	const H = 2 << 20
	sz := (nbytes + H - 1) / H * H
	b, err := syscall.Mmap(-1, 0, sz+H, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	off := (H - int(uintptr(unsafe.Pointer(&b[0])))%H) % H
	b = b[off : off+sz]
	adv := 15 // MADV_NOHUGEPAGE
	if huge {
		adv = 14 // MADV_HUGEPAGE
	}
	if err := syscall.Madvise(b, adv); err != nil {
		panic(err)
	}
	for i := range b {
		b[i] = 1
	}
	return b[:nbytes]
}

func median(x []float64) float64 {
	y := append([]float64(nil), x...)
	sort.Float64s(y)
	return y[len(y)/2]
}

var sink float64
var isink int

func freq(rounds int) {
	const n = 20_000_000 // 320M adds ≈ 0.1 s
	type fn struct {
		name string
		f    func(int) int
	}
	fs := []fn{{"scalar", chain}, {"ymm-mul", chainY}, {"scalar", chain}, {"zmm-mul", chainZ}, {"scalar", chain}}
	for r := 0; r < rounds; r++ {
		for _, f := range fs {
			t := time.Now()
			isink += f.f(n)
			d := time.Since(t)
			ns := float64(d.Nanoseconds()) / float64(n*16)
			fmt.Printf("freq round=%d %-8s %.4f ns/add  %.3f GHz\n", r, f.name, ns, 1/ns)
		}
	}
}

func latency(huge bool, sizes []int) {
	for _, sz := range sizes {
		b := alloc(sz, huge)
		lines := sz / 64
		perm := rand.Perm(lines)
		// cycle through perm: line perm[i] -> perm[i+1]
		for i := 0; i < lines; i++ {
			a := perm[i]
			nx := perm[(i+1)%lines]
			*(*uintptr)(unsafe.Pointer(&b[a*64])) = uintptr(unsafe.Pointer(&b[nx*64]))
		}
		p := unsafe.Pointer(&b[perm[0]*64])
		steps := 20_000_000
		if steps < 4*lines {
			steps = 4 * lines
		}
		for i := 0; i < lines; i++ { // warm
			p = *(*unsafe.Pointer)(p)
		}
		var res []float64
		for r := 0; r < 3; r++ {
			t := time.Now()
			for i := 0; i < steps; i++ {
				p = *(*unsafe.Pointer)(p)
			}
			res = append(res, float64(time.Since(t).Nanoseconds())/float64(steps))
		}
		isink += int(uintptr(p))
		fmt.Printf("lat huge=%v size=%dK %.2f ns\n", huge, sz>>10, median(res))
		syscall.Munmap(b[:cap(b)])
	}
}

func bw(huge bool, sizes []int) {
	for _, sz := range sizes {
		a := alloc(sz, huge)
		c := alloc(sz, huge)
		reps := (1 << 30) / sz
		if reps < 4 {
			reps = 4
		}
		meas := func(name string, f func()) {
			f()
			var res []float64
			for r := 0; r < 5; r++ {
				t := time.Now()
				f()
				res = append(res, float64(time.Since(t).Nanoseconds()))
			}
			ns := median(res)
			gbs := float64(sz) * float64(reps) / ns
			fmt.Printf("bw huge=%v size=%dK %-7s %.2f GB/s (%.2f B/ns per array byte)\n", huge, sz>>10, name, gbs, gbs)
		}
		meas("readZ", func() { sink += readZ(&a[0], sz, reps) })
		meas("readY", func() { sink += readY(&a[0], sz, reps) })
		meas("writeZ", func() { writeZ(&a[0], sz, reps) })
		meas("copyZ", func() { copyZ(&c[0], &a[0], sz, reps) })
		meas("copyY", func() { copyY(&c[0], &a[0], sz, reps) })
		syscall.Munmap(a[:cap(a)])
		syscall.Munmap(c[:cap(c)])
	}
}

func main() {
	mode := flag.String("mode", "all", "freq|lat|bw|all")
	flag.Parse()
	K := 1 << 10
	latSizes := []int{16 * K, 24 * K, 48 * K, 64 * K, 128 * K, 256 * K, 512 * K, 768 * K, 1024 * K, 1280 * K, 1536 * K, 2048 * K, 3072 * K, 4096 * K, 6144 * K, 8192 * K, 16384 * K, 32768 * K, 65536 * K, 262144 * K}
	bwSizes := []int{16 * K, 128 * K, 256 * K, 512 * K, 768 * K, 1024 * K, 1536 * K, 2048 * K, 4096 * K, 8192 * K, 16384 * K, 65536 * K, 262144 * K}
	if *mode == "freq" || *mode == "all" {
		freq(3)
	}
	if *mode == "lat" || *mode == "all" {
		latency(true, latSizes)
		latency(false, latSizes)
	}
	if *mode == "bw" || *mode == "all" {
		bw(true, bwSizes)
		bw(false, []int{1024 * K, 2048 * K, 16384 * K, 262144 * K})
	}
	fmt.Fprintln(os.Stderr, sink, isink)
}

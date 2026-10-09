package fft_test

// The measurement behind BENCHMARKS.md's Round 32: the smallest transforms
// timed on several heap placements of the same plan, so that a difference
// between two binaries can be told from a difference between two
// allocations. It uses the exported API only, so the same file builds
// against every release it is compared on.

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/go-fft/fft"
)

var (
	znPlace      = flag.String("zn.place", "", "rounds,ms,placements: run TestZnPlace")
	znPlaceSizes = flag.String("zn.sizes", "C32,C64,C128,C256,R256", "TestZnPlace's rows: C<n> complex, R<n> RFFT, I<n> IRFFT")
)

// znSink keeps the padding allocations alive for the whole test.
var znSink [][]byte

// znPad shifts every small size class's next allocation by a pseudo-random
// number of objects (placement 0: none), so that the plan built next lands
// elsewhere in the heap.
func znPad(j int) {
	if j == 0 {
		return
	}
	r := rand.New(rand.NewPCG(uint64(j), 32))
	for _, s := range []int{16, 32, 48, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2688, 3072, 4096, 5376, 6144, 8192} {
		for range r.IntN(8) {
			znSink = append(znSink, make([]byte, s))
		}
	}
}

func znMedian(x []float64) float64 {
	y := slices.Clone(x)
	slices.Sort(y)
	if len(y)%2 == 1 {
		return y[len(y)/2]
	}
	return (y[len(y)/2-1] + y[len(y)/2]) / 2
}

// TestZnPlace builds each row's plan at -zn.place's placements count of heap
// placements, then times every (row, placement) in rotated rounds and prints
// each one's median, and per row the median, minimum and maximum over the
// placements.
func TestZnPlace(t *testing.T) {
	if *znPlace == "" {
		t.Skip("set -zn.place=rounds,ms,placements")
	}
	var rounds, ms, places int
	if _, err := fmt.Sscanf(*znPlace, "%d,%d,%d", &rounds, &ms, &places); err != nil {
		t.Fatal(err)
	}
	type cell struct {
		row  string
		j    int
		addr string
		fn   func()
	}
	var cells []cell
	rows := strings.Split(*znPlaceSizes, ",")
	for j := range places {
		for _, row := range rows {
			znPad(j)
			n, err := strconv.Atoi(row[1:])
			if err != nil {
				t.Fatal(err)
			}
			var fn func()
			var a, b uintptr
			switch row[0] {
			case 'C':
				p := fft.NewPlan(n)
				src := make([]complex128, n)
				for i := range src {
					src[i] = complex(float64((i*7+1)%13)*0.1, float64(i*((i*3+2)%11))*0.1)
				}
				dst := make([]complex128, n)
				fn = func() { p.FFT(dst, src) }
				a, b = uintptr(unsafe.Pointer(&src[0])), uintptr(unsafe.Pointer(&dst[0]))
			case 'R':
				p := fft.NewRealPlan(n)
				src := make([]float64, n)
				for i := range src {
					src[i] = float64((i*7+1)%13) * 0.1
				}
				dst := make([]complex128, n/2+1)
				fn = func() { p.RFFT(dst, src) }
				a, b = uintptr(unsafe.Pointer(&src[0])), uintptr(unsafe.Pointer(&dst[0]))
			case 'I':
				p := fft.NewRealPlan(n)
				x := make([]float64, n)
				for i := range x {
					x[i] = float64((i*7+1)%13) * 0.1
				}
				spec := p.RFFT(make([]complex128, n/2+1), x)
				dst := make([]float64, n)
				fn = func() { p.IRFFT(dst, spec) }
				a, b = uintptr(unsafe.Pointer(&spec[0])), uintptr(unsafe.Pointer(&dst[0]))
			default:
				t.Fatalf("row %q", row)
			}
			fn()
			cells = append(cells, cell{row, j, fmt.Sprintf("src%%4K=%d dst%%4K=%d", a%4096, b%4096), fn})
		}
	}
	// Calibrate on the first cell of each row so every cell of a row runs
	// the same iteration count.
	iters := map[string]int{}
	for _, c := range cells {
		if _, ok := iters[c.row]; ok {
			continue
		}
		k := 1
		for {
			t0 := time.Now()
			for range k {
				c.fn()
			}
			if time.Since(t0) > time.Duration(ms)*time.Millisecond/4 {
				k *= 4
				break
			}
			k *= 2
		}
		iters[c.row] = k
	}
	times := make([][]float64, len(cells))
	for round := range rounds {
		for q := range cells {
			v := (q + round*7) % len(cells)
			k := iters[cells[v].row]
			t0 := time.Now()
			for range k {
				cells[v].fn()
			}
			times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(k))
		}
	}
	per := map[string][]float64{}
	for i, c := range cells {
		m := znMedian(times[i])
		per[c.row] = append(per[c.row], m)
		fmt.Printf("PLACE %-5s j=%-2d %7.1f  %s\n", c.row, c.j, m, c.addr)
	}
	for _, row := range rows {
		x := per[row]
		fmt.Printf("ROW %-5s median %7.1f  min %7.1f  max %7.1f  (%d placements)\n", row, znMedian(x), slices.Min(x), slices.Max(x), len(x))
	}
}

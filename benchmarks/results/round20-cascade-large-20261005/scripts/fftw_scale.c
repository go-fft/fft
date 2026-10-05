/* FFTW's own scaling on the host (go-fft Round 20): complex out-of-place,
 * FFTW_MEASURE, one thread, best of 7 batches of >= 50 ms; prints ns per
 * point and GFLOP/s, and FFTW's plan for the sizes named in PLANS. */
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int lo = 10, hi = 20;
    for (int e = lo; e <= hi; e++) {
        long n = 1L << e;
        fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n);
        fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
        for (long i = 0; i < n; i++)
            in[i] = ((i * 7 + 1) % 13) * 0.1 + I * (i * ((i * 3 + 2) % 11) * 0.1);
        fftw_execute(p);
        long it = 1;
        for (;;) {
            double t0 = now_s();
            for (long i = 0; i < it; i++) fftw_execute(p);
            if (now_s() - t0 > 0.05) break;
            it *= 2;
        }
        double best = 1e30;
        for (int b = 0; b < 7; b++) {
            double t0 = now_s();
            for (long i = 0; i < it; i++) fftw_execute(p);
            double d = (now_s() - t0) / it;
            if (d < best) best = d;
        }
        double ns = best * 1e9;
        printf("fftw n=%ld %.3f ns/pt %.2f GFLOP/s\n", n, ns / n, 5.0 * n * e / ns);
        if (e == 12 || e == 15 || e == 16 || e == 18 || e == 20) {
            printf("plan n=%ld: ", n);
            fflush(stdout);
            fftw_fprint_plan(p, stdout);
            printf("\n");
        }
        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);
    }
    return 0;
}

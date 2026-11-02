/* Prints FFTW's chosen plan (FFTW_MEASURE, same flags as the parity harness)
 * for the behind rows. Build: cc -O2 -I$P/include planprint.c $P/lib/libfftw3.a -lm */
#include <fftw3.h>
#include <stdio.h>

static void fl(fftw_plan p) {
    double a, m, f;
    fftw_flops(p, &a, &m, &f);
    printf("flops add %.0f mul %.0f fma %.0f\n", a, m, f);
}
#define fftw_print_plan(p) (fftw_print_plan(p), fl(p))

int main(void) {
    long cs[] = {65536};
    long rs[] = {256, 65536, 1048576};
    long sh[][2] = {{64, 64}, {128, 128}, {256, 256}, {1024, 1024}};
    for (int i = 0; i < 1; i++) {
        long n = cs[i];
        fftw_complex *in = fftw_malloc(16 * n), *out = fftw_malloc(16 * n);
        fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
        printf("== complex %ld\n", n); fftw_print_plan(p); printf("\n");
    }
    for (int i = 0; i < 3; i++) {
        long n = rs[i];
        double *in = fftw_malloc(8 * n);
        fftw_complex *out = fftw_malloc(16 * (n / 2 + 1));
        fftw_plan p = fftw_plan_dft_r2c_1d(n, in, out, FFTW_MEASURE);
        printf("== r2c %ld\n", n); fftw_print_plan(p); printf("\n");
    }
    for (int i = 0; i < 4; i++) {
        long r = sh[i][0], c = sh[i][1];
        fftw_complex *in = fftw_malloc(16 * r * c), *out = fftw_malloc(16 * r * c);
        fftw_plan p = fftw_plan_dft_2d(r, c, in, out, FFTW_FORWARD, FFTW_MEASURE);
        printf("== 2d %ldx%ld\n", r, c); fftw_print_plan(p); printf("\n");
    }
    return 0;
}

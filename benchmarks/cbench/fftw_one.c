/* Single-size native-FFTW runner for hardware-counter work (perf stat, IBS).
 *
 * Usage: fftw_one <n> <iters> [flags]   flags: m = FFTW_MEASURE (default),
 *        e = FFTW_EXHAUSTIVE-free ESTIMATE, p = FFTW_PATIENT
 *
 * Plans once, fills the same input as fftw_bench.c, runs <iters> executes
 * and prints "ns/op" on stdout. Counters taken around the whole process
 * include planning; take two runs with different <iters> and subtract
 * (perf stat on both) so only the steady-state executes remain.
 */
#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    long n = atol(argv[1]), iters = atol(argv[2]);
    unsigned fl = FFTW_MEASURE;
    if (argc > 3 && argv[3][0] == 'e') fl = FFTW_ESTIMATE;
    if (argc > 3 && argv[3][0] == 'p') fl = FFTW_PATIENT;
    fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, fl);
    if (getenv("FFTW_PRINT_PLAN")) {
        fftw_fprint_plan(p, stderr);
        fprintf(stderr, "\n");
    }
    for (long i = 0; i < n; i++)
        in[i] = ((i * 7 + 1) % 13) * 0.1 + I * (((i * 3 + 2) % 11) * 0.1);
    for (int i = 0; i < 100; i++) fftw_execute(p);
    double t0 = now_s();
    for (long i = 0; i < iters; i++) fftw_execute(p);
    double dt = now_s() - t0;
    printf("%ld %.1f ns/op\n", n, dt / iters * 1e9);
    return 0;
}

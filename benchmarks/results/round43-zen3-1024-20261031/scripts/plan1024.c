/* plan1024: print FFTW's plan for a complex double 1-D forward FFT of length n
   (out-of-place and in-place), for FFTW_MEASURE and FFTW_PATIENT, with timing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void one(int n, unsigned flags, const char *fname, int inplace) {
  fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * n);
  fftw_complex *out = inplace ? in : fftw_malloc(sizeof(fftw_complex) * n);
  fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, flags);
  for (int i = 0; i < n; i++) { in[i][0] = (double)(i % 7) - 3; in[i][1] = (double)(i % 5) - 2; }
  double best = 1e9;
  for (int r = 0; r < 31; r++) {
    int it = 20000;
    double t0 = now();
    for (int k = 0; k < it; k++) fftw_execute(p);
    double dt = (now() - t0) / it * 1e9;
    if (dt < best) best = dt;
  }
  double add, mul, fma;
  fftw_flops(p, &add, &mul, &fma);
  printf("== n=%d %s %s  best %.1f ns  flops add=%.0f mul=%.0f fma=%.0f\n", n, fname,
         inplace ? "in-place" : "out-of-place", best, add, mul, fma);
  fftw_print_plan(p);
  printf("\n");
  fftw_destroy_plan(p);
  if (!inplace) fftw_free(out);
  fftw_free(in);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 1024;
  one(n, FFTW_MEASURE, "MEASURE", 0);
  one(n, FFTW_MEASURE, "MEASURE", 1);
  one(n, FFTW_PATIENT, "PATIENT", 0);
  one(n, FFTW_PATIENT, "PATIENT", 1);
  return 0;
}

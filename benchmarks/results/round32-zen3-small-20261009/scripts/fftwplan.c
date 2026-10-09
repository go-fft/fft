/* fftwplan: FFTW_MEASURE plans for small c2c and r2c sizes, printed, and timed
   (best of 9 batches of ~50 ms, the same inputs as the go-fft harness). */
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double timeit(fftw_plan p){
  long it=1; double t0,t;
  for(;;){t0=now();for(long i=0;i<it;i++)fftw_execute(p);t=now()-t0;if(t>5e6)break;it*=2;}
  it=(long)(it*5e7/t)+1; double best=1e30;
  for(int b=0;b<9;b++){t0=now();for(long i=0;i<it;i++)fftw_execute(p);t=(now()-t0)/it;if(t<best)best=t;}
  return best;
}
int main(int argc,char**argv){
  int print = argc>1;
  int cs[]={32,64,128,256,512,1024};
  for(int k=0;k<6;k++){int n=cs[k];
    fftw_complex*in=fftw_malloc(sizeof(fftw_complex)*n),*out=fftw_malloc(sizeof(fftw_complex)*n);
    fftw_plan p=fftw_plan_dft_1d(n,in,out,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i<n;i++){in[i][0]=((i*7+1)%13)*0.1;in[i][1]=i*((i*3+2)%11)*0.1;}
    printf("ZN FFTW-C%d %.1f\n",n,timeit(p)); if(print){fftw_print_plan(p);printf("\n");}
    fftw_destroy_plan(p);fftw_free(in);fftw_free(out);}
  int rs[]={128,256,512,1024};
  for(int k=0;k<4;k++){int n=rs[k];
    double*in=fftw_malloc(sizeof(double)*n);fftw_complex*out=fftw_malloc(sizeof(fftw_complex)*(n/2+1));
    fftw_plan p=fftw_plan_dft_r2c_1d(n,in,out,FFTW_MEASURE);
    for(int i=0;i<n;i++)in[i]=((i*7+1)%13)*0.1;
    printf("ZN FFTW-R%d %.1f\n",n,timeit(p)); if(print){fftw_print_plan(p);printf("\n");}
    fftw_destroy_plan(p);fftw_free(in);fftw_free(out);}
  return 0;
}

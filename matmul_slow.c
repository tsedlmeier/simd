#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define N (int)pow(2,28)

__attribute__((optimize("no-tree-vectorize")))
void matmul(float *a, float *b, float *c, int n)
{
    for(int i=0; i<n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

void fill(float *a, float val, int n)
{
    for(int i=0; i<n; i++)
    {
        a[i] = val;
    }
}

int main()
{
    float *A = (float*) malloc(N*sizeof(float));
    float *B = (float*) malloc(N*sizeof(float));
    float *C = (float*) malloc(N*sizeof(float));
    fill(A, 2.0, N);
    fill(B, 3.0, N);
    fill(C, 0.0, N);

    clock_t start, end;
    double t_slow, t_fast;

    printf("Perform conventional Matrix-Mul (N=%d) \n", N);
    start = clock();
    matmul(A,B,C,N);
    end = clock();
    srand(time(NULL));
    int idx = rand() % N;
    printf("Check some random val: %f * %f = %f\n", A[idx], B[idx], C[idx]);
    t_slow = ((double)(end-start))/ CLOCKS_PER_SEC;
    printf("Time: %f s\n", t_slow);

    free(A);
    free(B);
    free(C);
    return 0;
}



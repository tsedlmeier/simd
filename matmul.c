#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define N (int)pow(2,28)

void matmul(float *a, float *b, float *c, int n)
{
    for(int i=0; i<n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

void matmul_sse2(float *a, float *b, float *c, int n)
{
    for(int i=0; i<n; i+= 4)
    {
        __m128 m1 = _mm_load_ps( &a[i] );
        __m128 m2 = _mm_load_ps( &b[i] );
        
        __m128 res = _mm_mul_ps( m1, m2 );

        _mm_store_ps( &c[i], res );
    }
}

void matmul_avx(float *a, float *b, float *c, int n)
{
    for(int i=0; i<n; i+= 8)
    {
        __m256 m1 = _mm256_load_ps( &a[i] );
        __m256 m2 = _mm256_load_ps( &b[i] );

        __m256 res = _mm256_mul_ps( m1, m2 );

        _mm256_store_ps( &c[i], res );
    }
}

void matmul_avx512(float *a, float *b, float *c, int n)
{
    for(int i=0; i<n; i+= 16)
    {
        __m512 m1 = _mm512_load_ps( &a[i] );
        __m512 m2 = _mm512_load_ps( &b[i] );

        __m512 res = _mm512_mul_ps( m1, m2 );

        _mm512_store_ps( &c[i], res );
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
    float *A = (float*) _mm_malloc(N*sizeof(float), 32);
    float *B = (float*) _mm_malloc(N*sizeof(float), 32);
    float *C = (float*) _mm_malloc(N*sizeof(float), 32);
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

    printf("Perform 128-Bit Version (N=%d)\n", N);
    start = clock();
    matmul_sse2(A,B,C,N);
    end = clock();
    idx = rand() % N;
    printf("Check some random val: %f * %f = %f\n", A[idx], B[idx], C[idx]);
    t_fast = ((double)(end-start))/ CLOCKS_PER_SEC;
    printf("Time: %f s\n", t_fast);

    printf("Perform 256-Bit Version (N=%d)\n", N);
    start = clock();
    matmul_avx(A,B,C,N);
    end = clock();
    idx = rand() % N;
    printf("Check some random val: %f * %f = %f\n", A[idx], B[idx], C[idx]);
    t_fast = ((double)(end-start))/ CLOCKS_PER_SEC;
    printf("Time: %f s\n", t_fast);

    printf("Perform 512-Bit Version (N=%d)\n", N);
    start = clock();
    matmul_avx512(A,B,C,N);
    end = clock();
    idx = rand() % N;
    printf("Check some random val: %f * %f = %f\n", A[idx], B[idx], C[idx]);
    t_fast = ((double)(end-start))/ CLOCKS_PER_SEC;
    printf("Time: %f s\n", t_fast);

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    return 0;
}



#ifndef HMM_H
#define HMM_H

#define N 2   // Number of states
#define M 2   // Number of observation symbols
#define T 5   // Length of observation sequence

void forward(int O[], double A[N][N], double B[N][M], double pi[N], double alpha[T][N]);
void backward(int O[], double A[N][N], double B[N][M], double beta[T][N]);
void baum_welch(int O[], double A[N][N], double B[N][M], double pi[N]);

#endif
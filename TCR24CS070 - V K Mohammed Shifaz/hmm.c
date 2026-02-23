#include <stdio.h>
#include "hmm.h"

void forward(int O[], double A[N][N], double B[N][M], double pi[N], double alpha[T][N]) {
    for(int i=0;i<N;i++)
        alpha[0][i] = pi[i] * B[i][O[0]];

    for(int t=1;t<T;t++) {
        for(int j=0;j<N;j++) {
            alpha[t][j] = 0;
            for(int i=0;i<N;i++)
                alpha[t][j] += alpha[t-1][i] * A[i][j];
            alpha[t][j] *= B[j][O[t]];
        }
    }
}

void backward(int O[], double A[N][N], double B[N][M], double beta[T][N]) {
    for(int i=0;i<N;i++)
        beta[T-1][i] = 1;

    for(int t=T-2;t>=0;t--) {
        for(int i=0;i<N;i++) {
            beta[t][i] = 0;
            for(int j=0;j<N;j++)
                beta[t][i] += A[i][j] * B[j][O[t+1]] * beta[t+1][j];
        }
    }
}

void baum_welch(int O[], double A[N][N], double B[N][M], double pi[N]) {
    double alpha[T][N], beta[T][N];
    double gamma[T][N], xi[T-1][N][N];
    int iterations = 5;

    for(int iter=0; iter<iterations; iter++) {

        forward(O, A, B, pi, alpha);
        backward(O, A, B, beta);

        for(int t=0;t<T-1;t++) {
            double denom = 0;
            for(int i=0;i<N;i++)
                denom += alpha[t][i] * beta[t][i];

            for(int i=0;i<N;i++) {
                gamma[t][i] = 0;
                for(int j=0;j<N;j++) {
                    xi[t][i][j] = (alpha[t][i] * A[i][j] *
                                   B[j][O[t+1]] * beta[t+1][j]) / denom;
                    gamma[t][i] += xi[t][i][j];
                }
            }
        }

        for(int i=0;i<N;i++)
            pi[i] = gamma[0][i];

        for(int i=0;i<N;i++) {
            for(int j=0;j<N;j++) {
                double num=0, den=0;
                for(int t=0;t<T-1;t++) {
                    num += xi[t][i][j];
                    den += gamma[t][i];
                }
                A[i][j] = num/den;
            }
        }

        for(int i=0;i<N;i++) {
            for(int k=0;k<M;k++) {
                double num=0, den=0;
                for(int t=0;t<T;t++) {
                    if(O[t]==k)
                        num += gamma[t][i];
                    den += gamma[t][i];
                }
                B[i][k] = num/den;
            }
        }
    }
}
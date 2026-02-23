#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 2     // Number of states
#define M 2     // Number of observation symbols
#define T 5     // Length of observation sequence
#define MAX_ITERS 100

double A[N][N] = {{0.6, 0.4}, {0.5, 0.5}};
double B[N][M] = {{0.7, 0.3}, {0.4, 0.6}};
double pi[N] = {0.8, 0.2};

int O[T] = {0, 1, 0, 1, 0};  // Observation sequence

double alpha[T][N];
double beta[T][N];
double gamma[T][N];
double xi[T][N][N];

void forward() {
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

void backward() {
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

void baum_welch() {
    for(int iter=0;iter<MAX_ITERS;iter++) {
        forward();
        backward();

        for(int t=0;t<T-1;t++) {
            double denom = 0;
            for(int i=0;i<N;i++)
                for(int j=0;j<N;j++)
                    denom += alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j];

            for(int i=0;i<N;i++) {
                gamma[t][i] = 0;
                for(int j=0;j<N;j++) {
                    xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]) / denom;
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
                A[i][j] = num / den;
            }
        }

        for(int i=0;i<N;i++) {
            for(int k=0;k<M;k++) {
                double num=0, den=0;
                for(int t=0;t<T;t++) {
                    if(O[t] == k)
                        num += gamma[t][i];
                    den += gamma[t][i];
                }
                B[i][k] = num / den;
            }
        }
    }
}

void print_model() {
    printf("\nUpdated Transition Matrix (A):\n");
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++)
            printf("%.4f ", A[i][j]);
        printf("\n");
    }

    printf("\nUpdated Emission Matrix (B):\n");
    for(int i=0;i<N;i++) {
        for(int j=0;j<M;j++)
            printf("%.4f ", B[i][j]);
        printf("\n");
    }
}

int main() {
    printf("Training HMM using Baum-Welch Algorithm...\n");
    baum_welch();
    print_model();
    return 0;
}

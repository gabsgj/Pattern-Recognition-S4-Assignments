#include <stdio.h>

#define N 2   // number of states
#define M 2   // number of observation symbols
#define T 6   // length of observation sequence
#define ITER 5

double A[N][N] = {{0.5, 0.5},
                  {0.4, 0.6}};

double B[N][M] = {{0.6, 0.4},
                  {0.3, 0.7}};

double pi[N] = {0.6, 0.4};

int O[T] = {0, 1, 0, 1, 1, 0};

double alpha[T][N], beta[T][N];
double gamma_val[T][N];
double xi[T-1][N][N];

void forward() {
    int t, i, j;
    for (i = 0; i < N; i++)
        alpha[0][i] = pi[i] * B[i][O[0]];

    for (t = 1; t < T; t++) {
        for (j = 0; j < N; j++) {
            alpha[t][j] = 0;
            for (i = 0; i < N; i++)
                alpha[t][j] += alpha[t-1][i] * A[i][j];
            alpha[t][j] *= B[j][O[t]];
        }
    }
}

void backward() {
    int t, i, j;
    for (i = 0; i < N; i++)
        beta[T-1][i] = 1;

    for (t = T-2; t >= 0; t--) {
        for (i = 0; i < N; i++) {
            beta[t][i] = 0;
            for (j = 0; j < N; j++)
                beta[t][i] += A[i][j] * B[j][O[t+1]] * beta[t+1][j];
        }
    }
}

void baum_welch() {
    int iter, t, i, j, k;
    double denom;

    for (iter = 0; iter < ITER; iter++) {
        forward();
        backward();

        for (t = 0; t < T-1; t++) {
            denom = 0;
            for (i = 0; i < N; i++)
                for (j = 0; j < N; j++)
                    denom += alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j];

            for (i = 0; i < N; i++) {
                gamma_val[t][i] = 0;
                for (j = 0; j < N; j++) {
                    xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]) / denom;
                    gamma_val[t][i] += xi[t][i][j];
                }
            }
        }

        for (i = 0; i < N; i++)
            pi[i] = gamma_val[0][i];

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                double num = 0, den = 0;
                for (t = 0; t < T-1; t++) {
                    num += xi[t][i][j];
                    den += gamma_val[t][i];
                }
                A[i][j] = num / den;
            }
        }

        for (i = 0; i < N; i++) {
            for (k = 0; k < M; k++) {
                double num = 0, den = 0;
                for (t = 0; t < T-1; t++) {
                    if (O[t] == k)
                        num += gamma_val[t][i];
                    den += gamma_val[t][i];
                }
                B[i][k] = num / den;
            }
        }
    }
}

void print_matrix() {
    int i, j;
    printf("\nTransition Matrix A:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
            printf("%0.3f ", A[i][j]);
        printf("\n");
    }

    printf("\nEmission Matrix B:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++)
            printf("%0.3f ", B[i][j]);
        printf("\n");
    }

    printf("\nInitial Probabilities pi:\n");
    for (i = 0; i < N; i++)
        printf("%0.3f ", pi[i]);
    printf("\n");
}

int main() {
    printf("HMM using Baum-Welch Algorithm\n");

    baum_welch();

    print_matrix();

    return 0;
}
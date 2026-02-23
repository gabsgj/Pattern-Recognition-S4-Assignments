#include <stdio.h>
#include "hmm.h"

int main() {

    int O[T] = {0,1,1,0,1}; // W,H,H,W,H

    double A[N][N] = {{0.7,0.3},
                      {0.4,0.6}};

    double B[N][M] = {{0.1,0.9},
                      {0.6,0.4}};

    double pi[N] = {0.6,0.4};

    printf("Initial Transition Matrix:\n");
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++)
            printf("%.3f ",A[i][j]);
        printf("\n");
    }

    baum_welch(O,A,B,pi);

    printf("\nUpdated Transition Matrix:\n");
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++)
            printf("%.3f ",A[i][j]);
        printf("\n");
    }

    printf("\nUpdated Emission Matrix:\n");
    for(int i=0;i<N;i++) {
        for(int j=0;j<M;j++)
            printf("%.3f ",B[i][j]);
        printf("\n");
    }

    return 0;
}
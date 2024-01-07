#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <locale.h>
#include <mpi.h>

#define SIZE 1024

void create_matrix(double* m1, double* m2, int N) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            m1[i * N + j] = m2[i * N + j] = i + j;
        }
    }
}

void init(double* m1, double* m2, double* m3, double* result, int N) {
    int i;
    create_matrix(m1, m2, N);
    for (i = 0; i < N * N; i++) {
        m3[i] = 0;
        result[i] = 0;
    }
}

void multiply_matrices(double* m1, double* m2, double* m3, int N, int nRunk, int nSize) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0 + nRunk; j < N; j += nSize) {
            for (k = 0; k < N; k++) {
                m3[i * N + j] += m1[i * N + k] * m2[k * N + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    double* m1;
    double* m2;
    double* m3;
    double* result = NULL;
    int N = SIZE;
    int nRunk, nSize, c;
    double t1, t2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nRunk);
    MPI_Comm_size(MPI_COMM_WORLD, &nSize);

    if (nRunk == 0) {
        m1 = (double*)malloc(N * N * sizeof(double));
        m2 = (double*)malloc(N * N * sizeof(double));
        m3 = (double*)malloc(N * N * sizeof(double));
        result = (double*)malloc(N * N * sizeof(double));

        init(m1, m2, m3, result, N);

        t1 = MPI_Wtime();

        for (c = 1; c < nSize; c++) {
            MPI_Send(&N, 1, MPI_INT, c, 0, MPI_COMM_WORLD);
            MPI_Send(m1, N * N, MPI_DOUBLE, c, 1, MPI_COMM_WORLD);
            MPI_Send(m2, N * N, MPI_DOUBLE, c, 2, MPI_COMM_WORLD);
            MPI_Send(m3, N * N, MPI_DOUBLE, c, 3, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        m1 = (double*)malloc(N * N * sizeof(double));
        m2 = (double*)malloc(N * N * sizeof(double));
        m3 = (double*)malloc(N * N * sizeof(double));

        MPI_Recv(m1, N * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(m2, N * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(m3, N * N, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    multiply_matrices(m1, m2, m3, N, nRunk, nSize);
    MPI_Reduce(m3, result, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (nRunk == 0) {
        t2 = MPI_Wtime();
        printf("Время: %f сек.\n", t2 - t1);
        free(m1);
        free(m2);
        free(m3);
        free(result);
    }

    MPI_Finalize();
    return 0;
}

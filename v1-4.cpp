#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int main (int argc, char **argv){
    int comm_size, prank;
    int N = 10;
    int v[N], m[N][N];
    int local_min;
    int local_max;
    int pieces;

    srand(time(0));

    for(int i = 0; i < N; i++){
        v[i] = 1;//rand()%20;
        for(int j = 0; j < N; j++){
            m[i][j] = i;//rand()%20;
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    pieces = ceil(float(N)/comm_size);
    int total[pieces*comm_size];
    int local_sum[pieces];

    local_min = prank * pieces;
    
    local_max = local_min + pieces;

    for(int i = local_min; i < local_max; i++){
        local_sum[i - local_min] = 0;
        if(i < N){
            for(int j = 0; j < N; j++){
                local_sum[i - local_min] += v[j] * m[i][j];
            }
        }
    }

    MPI_Gather(local_sum, pieces, MPI_INT, total, pieces, MPI_INT, 0, MPI_COMM_WORLD);

    if(prank == 0){
        printf("Product of v and m is: [%d", total[0]);
        for(int i = 1; i < N; i++){
            printf(", %d", total[i]);
        }
        printf("]\n");
    }

    MPI_Finalize();
}
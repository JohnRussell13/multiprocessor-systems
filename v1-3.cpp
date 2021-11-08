#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int main (int argc, char **argv){
    int comm_size, prank;
    int N = 3;
    int v1[N], v2[N];
    int local_min;
    int local_max;
    int local_sum = 0;
    int total;
    int pieces;

    srand(time(0));

    for(int i = 0; i < N; i++){
        v1[i] = rand()%20;
        v2[i] = rand()%20;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    pieces = ceil(float(N)/comm_size);

    local_min = prank * pieces;
    if(prank != comm_size - 1){
        local_max = local_min + pieces;
    }
    else{
        local_max = N;
    }

    if(local_min < N){
        for(int i = local_min; i < local_max; i++){
            local_sum += v1[i] * v2[i];
        }
    }

    MPI_Reduce(&local_sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(prank == 0){
        printf("Dot product of v1 and v2 is: [%d", v1[0]);
        for(int i = 1; i < N; i++){
            printf(", %d", v1[i]);
        }
        printf("] . [%d", v2[0]);
        for(int i = 1; i < N; i++){
            printf(", %d", v2[i]);
        }
        printf("] = %d\n", total);
    }

    MPI_Finalize();
}
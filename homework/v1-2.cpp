#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main (int argc, char **argv){
    int comm_size, prank;
    int N = atoi(argv[argc - 1]);
    int local_min;
    int local_max;
    int local_sum = 0, total;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    local_min = prank * ceil(N/comm_size);
    if(prank != comm_size - 1){
        local_max = local_min + ceil(N/comm_size);
    }
    else{
        local_max = N + 1;
    }

    for(int i = local_min; i < local_max; i++){
        local_sum += i;
    }

    MPI_Reduce(&local_sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(prank == 0){
        printf("Total sum of the first %d natural numbers is: %d\n", N, total);
    }

    /*if(prank != 0){
        MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else{
        total = local_sum;
        for(int i = 1; i < comm_size; i++){
            MPI_Recv(&local_sum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += local_sum;
        }
        printf("Total sum of the first %d natural numbers is: %d\n", N, total);
    }*/

    MPI_Finalize();
}
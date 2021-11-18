#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char **argv){
    int comm_size, prank;
    int pkg;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    
    //int r[prank];
    int *r = (int*) malloc(prank * sizeof(int));

    srand(time(0) + prank);
    pkg = 10*prank + rand()%10;

    for (int i = 0; i < comm_size; i++){
        if(i != prank){
            MPI_Send(&pkg, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < comm_size; i++){
        if(i != prank){
            MPI_Recv(&r[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    printf("Process %d recieved: ", prank);
    for (int i = 0; i < comm_size; i++){
        if(i != prank){
            printf("%d ", r[i]);
        }
    }
    printf("\n");
    free(r);
    r = NULL;

    MPI_Finalize();
}

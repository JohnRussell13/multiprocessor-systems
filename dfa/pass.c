#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define size1 100
#define size2 100
#define in_len 14*14
#define len_dim 3

void print_arr(double *arr, int n){
    for(int i = 0; i < n; i++){
        printf("%lf ", arr[i]);
    }
    printf("\n");
}

int largest(double *arr, int n){
    int ind = 0;
    double max = arr[0];
    for(int i = 0; i < n; i++){
        if(max < arr[i]){
            ind = i;
            max = arr[i];
        }
    }
    return ind;
}

int main (int argc, char **argv){
    srand(time(NULL));
    /* MPI INIT */
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size--; // rank == size -> final part of the network
    int dim[len_dim+1] = {in_len, 10, 10, 10};

    int per_layer = size/len_dim; // how many processors per layer
    int layer; // layer map
    int layer_dim;
    int prev_layer_dim = dim[len_dim]; // for rank == size
    int layer_part_size = 0;
    int prev_layer_part_size;
    if(rank != size){
        layer = rank/per_layer; // layer map
        layer_dim = dim[layer+1];
        prev_layer_dim = dim[layer];
        layer_part_size = layer_dim/per_layer;
        prev_layer_part_size = prev_layer_dim/per_layer;
    }
    
    /* LOAD DATA */
    FILE *fp_xtr, *fp_ytr, *fp_xts, *fp_yts;
    double x_train[size1][in_len], y_train[size1], x_test[size2][in_len], y_test[size2];
    fp_xtr = fopen("img/x_train.txt", "r");
    fp_ytr = fopen("img/y_train.txt", "r");
    fp_xts = fopen("img/x_test.txt", "r");
    fp_yts = fopen("img/y_test.txt", "r");
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < in_len; j++){
            fscanf(fp_xtr, "%lf", &x_train[i][j]);
            fscanf(fp_xts, "%lf", &x_test[i][j]);
        }
        fscanf(fp_ytr, "%lf", &y_train[i]);
        fscanf(fp_yts, "%lf", &y_test[i]);
    }
    fclose(fp_xtr);
    fclose(fp_ytr);
    fclose(fp_xts);
    fclose(fp_yts);

    /* INIT */
    //double weights[layer_part_size][prev_layer_dim+1]; //dyn alloc
    //double *weights = (double*) malloc(layer_part_size * (prev_layer_dim+1) * sizeof(double));
    double **weights = (double**) malloc(layer_part_size * sizeof(double *));
    if(rank != size){
        for(int i = 0; i < layer_part_size; i++){
            weights[i] = (double*) malloc((prev_layer_dim+1) * sizeof(weights));
        }

        for(int i = 0; i < layer_part_size; i++){
            for(int j = 0; j < prev_layer_dim+1; j++){
                weights[i][j] = (double(rand())/RAND_MAX - 0.5) / 4; // random
            }
        }
    }
    //double net_in[prev_layer_dim]; // dyn_alloc
    double *net_in = (double*) malloc(prev_layer_dim * sizeof(double));
    
    /* TRAIN */
    int perc = 100;
    int per = size1*perc/100;
    int t = 0, p = 0;
    int tag = 0; // TODO image count?

    for(int i = 0; i < per; i++){
        if(rank != size){
            // INIT
            if(layer == 0){
                for(int j = 0; j < in_len; j++){ // in_len = dim[0]
                    net_in[j] = x_train[i][j];
                }
            }
            else{
                for(int j = 0; j < per_layer; j++){
                    MPI_Recv(&net_in[j*prev_layer_part_size], prev_layer_part_size, MPI_DOUBLE, per_layer*(layer-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            double *net_out = (double*) malloc(layer_part_size * sizeof(double));

            for(int j = 0; j < layer_part_size; j++){
                net_out[j] = weights[j][layer_dim]; // bias
                for(int k = 0; k < prev_layer_dim; k++){
                    net_out[j] += weights[j][k] * net_in[k]; // weights * previous_layer_output
                }
            }

            if(layer < len_dim - 1){
                // ACTIVATION
                for(int j = 0; j < layer_part_size; j++){
                    if(net_out[j] < 0){
                        net_out[j] = 0; //relu
                    }
                }
                // FEED FORWARD 
                for(int j = 0; j < per_layer; j++){
                    MPI_Send(net_out, layer_part_size, MPI_DOUBLE, per_layer*(layer+1)+j, tag, MPI_COMM_WORLD);
                }
            }
            else{
                MPI_Send(net_out, layer_part_size, MPI_DOUBLE, size, tag, MPI_COMM_WORLD);
            }
            //TODO SOFTMAX
        }
        else{
            for(int j = 0; j < per_layer; j++){
                MPI_Recv(&net_in[j*dim[len_dim]/per_layer], dim[len_dim]/per_layer, MPI_DOUBLE, per_layer*(len_dim-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if(i % (per/10) == 0){
                printf("Testing %d..\n", p*10);
                p++;
            }
            if(largest(net_in, dim[len_dim]) == y_train[i]){
                t++;
            }
        }
    }
    if(rank == size){
        printf("Accuracy: %d/%d\n", t, per);
    }

    MPI_Finalize();
}


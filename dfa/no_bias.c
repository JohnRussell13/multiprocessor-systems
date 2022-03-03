#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define size1 2000 // NUMBER OF TRAINING IMAGES
#define size2 2000 // NUMBER OF TESTING IMAGES
#define in_len 14*14 // INPUT SIZE
#define len_dim 3 // NUMBER OF NEURAL LAYERS (DEEP + 1)

#define tag 0

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
    int dim[len_dim+1] = {in_len, 100, 100, 10};

    srand(time(NULL));

    /* MPI INIT */
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size--; // rank == size -> final part of the network

    int per_layer = size/len_dim; // how many processors per layer
    int layer; // layer map
    int layer_dim;
    int prev_layer_dim = dim[len_dim]; // for rank == size
    int layer_part_size = 0;
    int prev_layer_part_size;
    int last = dim[len_dim];
    int part_of_layer;
    if(rank != size){
        layer = rank/per_layer; // layer map
        layer_dim = dim[layer+1];
        prev_layer_dim = dim[layer];
        layer_part_size = layer_dim/per_layer;
        prev_layer_part_size = prev_layer_dim/per_layer;
        last = 0;
        part_of_layer = rank - layer*per_layer;
    }
    
    /* LOAD DATA */
    FILE *fp_xtr, *fp_ytr, *fp_xts, *fp_yts;
    double x_train[size1][in_len], x_test[size2][in_len];
    int y_train[size1], y_test[size2];
    fp_xtr = fopen("img/x_train.txt", "r");
    fp_ytr = fopen("img/y_train.txt", "r");
    fp_xts = fopen("img/x_test.txt", "r");
    fp_yts = fopen("img/y_test.txt", "r");
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < in_len; j++){
            fscanf(fp_xtr, "%lf", &x_train[i][j]);
        }
        fscanf(fp_ytr, "%d", &y_train[i]);
    }
    for(int i = 0; i < size2; i++){
        for(int j = 0; j < in_len; j++){
            fscanf(fp_xts, "%lf", &x_test[i][j]);
        }
        fscanf(fp_yts, "%d", &y_test[i]);
    }
    fclose(fp_xtr);
    fclose(fp_ytr);
    fclose(fp_xts);
    fclose(fp_yts);

    /* INIT */
    double **weights = (double**) malloc(layer_part_size * sizeof(double *));
    if(rank != size){
        for(int i = 0; i < layer_part_size; i++){
            weights[i] = (double*) malloc(prev_layer_dim * sizeof(weights));
        }
        for(int i = 0; i < layer_part_size; i++){
            for(int j = 0; j < prev_layer_dim; j++){
                weights[i][j] = (double(rand())/RAND_MAX - 0.5) / 4; // random
            }
        }
    }
    double **B = (double**) malloc(layer_part_size * sizeof(double *));
    if(rank != size){
        for(int i = 0; i < layer_part_size; i++){
            B[i] = (double*) malloc((dim[len_dim]) * sizeof(B));
        }
        for(int i = 0; i < layer_part_size; i++){
            for(int j = 0; j < dim[len_dim]; j++){
                B[i][j] = (double(rand())/RAND_MAX); // random
                B[i][j] = B[i][j] / sqrt(layer_dim);
            }
        }
    }
    double *da = (double*) malloc(layer_part_size * sizeof(double));
    double *net_in = (double*) malloc(prev_layer_dim * sizeof(double));
    double *net_out = (double*) malloc(layer_part_size * sizeof(double));
    int *arr = (int*) malloc(last * sizeof(int));
    double *delta = (double*) malloc(dim[len_dim] * sizeof(double));
    double *res_exp = (double*) malloc(last * sizeof(double));
    double sum;
    
    /* TRAIN */
    double val_split = 0.2; // PART USED FOR TESTING
    int epochs = 30; // NUMBER OF EPOCHS
    double learn_rate = 0.05; //LEARNING COEF.

    int tot;
    double ce;
    int b_tot;
    double b_ce;
    int part = int(size1*(1-val_split));

    // START COUNTER
    clock_t timer;
    if(rank == size){
        timer = clock();
    }
    
    for(int epoch = 0; epoch < epochs; epoch++){
        for(int i = 0; i < part; i++){
            /* PASS */
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

                for(int j = 0; j < layer_part_size; j++){
                    net_out[j] = 0; // bias
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
            }
            else{
                for(int j = 0; j < per_layer; j++){
                    MPI_Recv(&net_in[j*dim[len_dim]/per_layer], dim[len_dim]/per_layer, MPI_DOUBLE, per_layer*(len_dim-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last; j++){
                    res_exp[j] = exp(net_in[j]);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last; j++){
                    net_in[j] = res_exp[j]/sum;
                    if(j == y_train[i]){
                        delta[j] = net_in[j] - 1;
                    }
                    else{
                        delta[j] = net_in[j];
                    }
                }
            }
            MPI_Bcast(delta, dim[len_dim], MPI_DOUBLE, size, MPI_COMM_WORLD);

            /* UPDATE */
            if(rank != size){
                if(layer < len_dim-1){
                    for(int j = 0; j < layer_part_size; j++){
                        da[j] = 0;
                        for(int k = 0; k < dim[len_dim]; k++){
                            da[j] += B[j][k] * delta[k];
                        }
                    }
                }
                else{
                    for(int j = 0; j < layer_part_size; j++){
                        da[j] = delta[part_of_layer*layer_part_size + j];
                    }
                }
                for(int j = 0; j < layer_part_size; j++){
                    for(int k = 0; k < prev_layer_dim; k++){
                        weights[j][k] += -learn_rate * da[j]*net_in[k];
                    }
                }
            }
        }

        tot = 0;
        ce = 0;
        for(int i = part; i < size1; i++){
            /* PASS */
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
                    net_out[j] = 0; // bias
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
            }
            else{
                for(int j = 0; j < per_layer; j++){
                    MPI_Recv(&net_in[j*dim[len_dim]/per_layer], dim[len_dim]/per_layer, MPI_DOUBLE, per_layer*(len_dim-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last; j++){
                    res_exp[j] = exp(net_in[j]);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last; j++){
                    net_in[j] = res_exp[j]/sum;
                }
                ce -= log(net_in[y_train[i]]);
                if(largest(net_in, dim[len_dim]) == y_train[i]){
                    tot++;
                }
            }
        }
        
        b_tot = 0;
        b_ce = 0;
        for(int i = 0; i < size2; i++){
            /* PASS */
            if(rank != size){
                // INIT
                if(layer == 0){
                    for(int j = 0; j < in_len; j++){ // in_len = dim[0]
                        net_in[j] = x_test[i][j];
                    }
                }
                else{
                    for(int j = 0; j < per_layer; j++){
                        MPI_Recv(&net_in[j*prev_layer_part_size], prev_layer_part_size, MPI_DOUBLE, per_layer*(layer-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                double *net_out = (double*) malloc(layer_part_size * sizeof(double));

                for(int j = 0; j < layer_part_size; j++){
                    net_out[j] = 0; // bias
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
            }
            else{
                for(int j = 0; j < per_layer; j++){
                    MPI_Recv(&net_in[j*dim[len_dim]/per_layer], dim[len_dim]/per_layer, MPI_DOUBLE, per_layer*(len_dim-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last; j++){
                    res_exp[j] = exp(net_in[j]);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last; j++){
                    net_in[j] = res_exp[j]/sum;
                }
                b_ce -= log(net_in[y_test[i]]);
                if(largest(net_in, dim[len_dim]) == y_test[i]){
                    b_tot++;
                }
            }
        }

        if(rank == size){
            printf("Epoch %d/%d: ", epoch+1, epochs);
            printf("Seen - Err: %.5lf; ", ce/(size1-part));
            printf("Acc: %.5lf; ", double(tot)/(size1-part));
            printf("Unseen - Err: %.5lf; ", b_ce/size2);
            printf("Acc: %.5lf.\n", double(b_tot)/size2);
        }
    }

    // END COUNTER
    if(rank == size){
        timer = clock() - timer;
        double time_taken = ((double)timer)/CLOCKS_PER_SEC;
        printf("Training finished in %.2lfs.\n", time_taken);
    }
    

    MPI_Finalize();
}
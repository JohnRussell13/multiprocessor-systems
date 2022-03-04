#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define size1 10000 // NUMBER OF TRAINING IMAGES
#define size2 10000 // NUMBER OF TESTING IMAGES
#define in_len 14*14 // INPUT SIZE

#define tag 0

void fprint_arri(FILE *fp, int *arr, int n){
    for(int i = 0; i < n; i++){
        fprintf(fp, "%d ", arr[i]);
    }
    fprintf(fp, "\n");
}

void fprint_arrd(FILE *fp, double *arr, int n){
    for(int i = 0; i < n; i++){
        fprintf(fp, "%lf ", arr[i]);
    }
    fprintf(fp, "\n");
}

void print_arri(int *arr, int n){
    for(int i = 0; i < n; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void print_arrd(double *arr, int n){
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
    /* NETWORK PARAMS */
    int len_dim = argc-1;
    int *dim = (int*) malloc((len_dim + 1) * sizeof(int));
    dim[0] = in_len;
    for(int i = 1; i <= len_dim; i++){
        dim[i] = atoi(argv[i]);
    }

    double val_split = 0.2; // PART USED FOR TESTING
    int epochs = 100; // NUMBER OF EPOCHS
    double learn_rate = 0.05; // LEARNING COEF.
    int batch_size = 1; // MAXIMAL SIZE OF ONE BATCH
    //double lambda = 0.0000001; // L2 REG
    
    learn_rate = learn_rate/len_dim; // STABILIZING DEEP NETWORKS


    /* MPI INIT */
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size--; // rank == size -> final part of the network

    FILE *fp;
    char name[30];
    sprintf(name, "log/%d.txt", rank);
    fp = fopen(name, "w");

    int last_size = dim[len_dim];
    int layer = -1;
    int size_prev_layer = last_size;
    int size_this_layer = -1;
    int num_proc_prev_layer = 0;
    int num_proc_this_layer = 0;
    int num_proc_next_layer = 0;
    int part_of_layer = -1;
    int this_block_size = 0;
    int this_layer_size = 0;

    if(rank != size){
        layer = rank % len_dim;
        this_layer_size = dim[layer+1];
        part_of_layer = rank / len_dim;
        size_prev_layer = dim[layer];
        size_this_layer = dim[layer+1];
        num_proc_this_layer = size/len_dim + ((size%len_dim) > layer);
        this_block_size = this_layer_size/num_proc_this_layer + ((this_layer_size%num_proc_this_layer) > part_of_layer);

        if(layer != len_dim && layer != 0){
            num_proc_prev_layer = size/len_dim + ((size%len_dim) > (layer-1));
            num_proc_next_layer = size/len_dim + ((size%len_dim) > (layer+1));
        }
        if(layer == 0){
            num_proc_next_layer = size/len_dim + ((size%len_dim) > (layer+1));
        }
        if(layer == len_dim){
            num_proc_prev_layer = size/len_dim + ((size%len_dim) > (layer-1));
        }
    }
    else{
        num_proc_prev_layer = size/len_dim + ((size%len_dim) > (len_dim-1));
        size_prev_layer = dim[len_dim];
    }

    int *size_blocks_prev_layer = (int*) malloc(num_proc_prev_layer * sizeof(int));
    for(int i = 0; i < num_proc_prev_layer; i++){
        size_blocks_prev_layer[i] = size_prev_layer/num_proc_prev_layer + ((size_prev_layer%num_proc_prev_layer) > i);
    }

    int *address_blocks_prev_layer = (int*) malloc(num_proc_prev_layer * sizeof(int));
    int *address_blocks_this_layer = (int*) malloc(num_proc_this_layer * sizeof(int));
    address_blocks_prev_layer[0] = 0;
    for(int i = 1; i < num_proc_prev_layer; i++){
        address_blocks_prev_layer[i] = address_blocks_prev_layer[i-1] + size_blocks_prev_layer[i-1];
    }
    address_blocks_this_layer[0] = 0;
    for(int i = 1; i < num_proc_this_layer; i++){
        address_blocks_this_layer[i] = address_blocks_this_layer[i-1] + size_this_layer/num_proc_this_layer + ((size_this_layer%num_proc_this_layer) > (i-1));
    }

    srand(rank + time(NULL));
    // ALLOCATE MEMORY ONLY WHERE NEEDED
    int last_size_m = 0;
    if(rank == size){
        last_size_m = last_size;
    }
    double **weights = (double**) malloc(this_block_size * sizeof(double *));
    if(rank != size){
        for(int i = 0; i < this_block_size; i++){
            weights[i] = (double*) malloc((size_prev_layer+1) * sizeof(weights));
        }
        for(int i = 0; i < this_block_size; i++){
            for(int j = 0; j < size_prev_layer+1; j++){
                weights[i][j] = (double(rand())/RAND_MAX - 0.5) / 4; // W ~ U(-0.125, 0.125)
            }
        }
    }
    double temp_rand;
    double **B = (double**) malloc(this_block_size << 1 * sizeof(double *));
    if(rank != size){
        for(int i = 0; i < this_block_size << 1; i++){
            B[i] = (double*) malloc(last_size * sizeof(B));
        }
        for(int i = 0; i < this_block_size << 1; i++){
            for(int j = 0; j < last_size; j++){
                temp_rand = (double(rand()+1)/RAND_MAX); // random
                B[i][j] = temp_rand*temp_rand/sqrt(size_prev_layer); //sqrt(size_prev_layer)

                // B ~ N(0,1) * 0.05    
                // temp_rand = sqrt(-log((double(rand()+1)/RAND_MAX))) * cos(2*M_PI*(double(rand()+1)/RAND_MAX));
                // B[i][j] = temp_rand * 0.05;
            }
        }
    }
    double *da = (double*) malloc(this_block_size << 1 * sizeof(double));
    double *net_in = (double*) malloc(size_prev_layer * sizeof(double));
    double *net_out = (double*) malloc(this_block_size * sizeof(double));
    double *delta = (double*) malloc(last_size * sizeof(double));
    for(int i = 0; i < last_size; i++){
        delta[i] = 0;
    }
    double *res_exp = (double*) malloc(last_size_m * sizeof(double));
    double sum;

    
    /* LOAD DATA */
    // ALLOCATE MEMORY ONLY WHERE NEEDED
    int l0_s1 = 0;
    int l0_s2 = 0;
    if(layer == 0){
        l0_s1 = size1;
        l0_s2 = size2;
    }
    int ls_s1 = 0;
    int ls_s2 = 0;
    if(rank == size){
        ls_s1 = size1;
        ls_s2 = size2;
    }
    FILE *fp_xtr, *fp_ytr, *fp_xts, *fp_yts;
    fp_xtr = fopen("img/x_train.txt", "r");
    fp_ytr = fopen("img/y_train.txt", "r");
    fp_xts = fopen("img/x_test.txt", "r");
    fp_yts = fopen("img/y_test.txt", "r");

    double **x_train = (double**) malloc(l0_s1 * sizeof(double *));
    int *y_train = (int*) malloc(ls_s1 * sizeof(int));
    double **x_test = (double**) malloc(l0_s2 * sizeof(double *));
    int *y_test = (int*) malloc(ls_s2 * sizeof(int));

    if(layer == 0){
        for(int i = 0; i < size1; i++){
            x_train[i] = (double*) malloc(in_len * sizeof(x_train));
        }
        for(int i = 0; i < size2; i++){
            x_test[i] = (double*) malloc(in_len * sizeof(x_test));
        }

        for(int i = 0; i < size1; i++){
            for(int j = 0; j < in_len; j++){
                fscanf(fp_xtr, "%lf", &x_train[i][j]);
            }
        }
        for(int i = 0; i < size2; i++){
            for(int j = 0; j < in_len; j++){
                fscanf(fp_xts, "%lf", &x_test[i][j]);
            }
        }
    }
    if(rank == size){
        for(int i = 0; i < size1; i++){
            fscanf(fp_ytr, "%d", &y_train[i]);
        }
        for(int i = 0; i < size2; i++){
            fscanf(fp_yts, "%d", &y_test[i]);
        }
    }

    fclose(fp_xtr);
    fclose(fp_ytr);
    fclose(fp_xts);
    fclose(fp_yts);
    
    /* TRAIN */
    int tot;
    double ce;
    int b_tot;
    double b_ce;
    int part = int(size1*(1-val_split));
    int div;

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
                    for(int j = 0; j < in_len; j++){
                        net_in[j] = x_train[i][j];
                    }
                }
                else{
                    for(int j = 0; j < num_proc_prev_layer; j++){
                        MPI_Recv(&net_in[address_blocks_prev_layer[j]], size_blocks_prev_layer[j], MPI_DOUBLE, j*len_dim+(layer-1), tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                for(int j = 0; j < this_block_size; j++){
                    net_out[j] = weights[j][size_this_layer]; // bias
                    for(int k = 0; k < size_prev_layer; k++){
                        net_out[j] += weights[j][k] * net_in[k]; // weights * previous_layer_output
                    }
                }

                if(layer < len_dim - 1){
                    // ACTIVATION
                    for(int j = 0; j < this_block_size; j++){
                        if(net_out[j] < 0){
                            net_out[j] = 0; //relu
                        }
                    }
                    
                    // FEED FORWARD 
                    for(int j = 0; j < num_proc_next_layer; j++){
                        MPI_Send(net_out, this_block_size, MPI_DOUBLE, j*len_dim + (layer+1), tag, MPI_COMM_WORLD);
                    }
                }
                else{
                    MPI_Send(net_out, this_block_size, MPI_DOUBLE, size, tag, MPI_COMM_WORLD);
                }
            }
            else{
                for(int j = 0; j < num_proc_prev_layer; j++){
                    MPI_Recv(&net_in[address_blocks_prev_layer[j]], size_blocks_prev_layer[j], MPI_DOUBLE, j*len_dim+(len_dim-1), tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last_size; j++){
                    res_exp[j] = exp(net_in[j]);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last_size; j++){
                    net_in[j] = res_exp[j]/sum;
                    if(j == y_train[i]){
                        delta[j] += net_in[j] - 1;
                    }
                    else{
                        delta[j] += net_in[j];
                    }
                }
            }

            /* UPDATE */
            if((i+1) % batch_size == 0 || (i+1) == part){
                if(rank == size){
                    if((i+1) % batch_size == 0){
                        div = batch_size;
                    }
                    else{
                        div = (i+1) % batch_size;
                    }
                    for(int j = 0; j < last_size; j++){
                        delta[j] = delta[j] / div;
                    }
                }
                MPI_Bcast(delta, last_size, MPI_DOUBLE, size, MPI_COMM_WORLD);

                if(rank != size){
                    if(layer < len_dim-1){
                        for(int j = 0; j < this_block_size << 1; j++){
                            da[j] = 0;
                            if(net_out != 0){ // da = (B . delta) * act'(a)
                                for(int k = 0; k < last_size; k++){
                                    da[j] += B[j][k] * delta[k];
                                }
                            }
                        }
                    }
                    else{
                        for(int j = 0; j < this_block_size; j++){
                            da[j] = delta[address_blocks_this_layer[part_of_layer] + j];
                            da[this_block_size + j] = delta[address_blocks_this_layer[part_of_layer] + j];
                        }
                    }
                    for(int j = 0; j < this_block_size; j++){
                        for(int k = 0; k < size_prev_layer; k++){
                            weights[j][k] -= learn_rate * da[j] * net_in[k];
                            //weights[j][k] -= (learn_rate * da[j] + lambda*weights[j][k]) * net_in[k]; // L2 REG
                        }
                    }
                    for(int j = 0; j < this_block_size; j++){
                        weights[j][size_prev_layer] -= learn_rate * da[this_block_size + j];
                    }
                }
                else{
                    for(int j = 0; j < last_size; j++){
                        delta[j] = 0;
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
                    for(int j = 0; j < num_proc_prev_layer; j++){
                        MPI_Recv(&net_in[address_blocks_prev_layer[j]], size_blocks_prev_layer[j], MPI_DOUBLE, j*len_dim+(layer-1), tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                double *net_out = (double*) malloc(this_block_size * sizeof(double));

                for(int j = 0; j < this_block_size; j++){
                    net_out[j] = weights[j][this_layer_size]; // bias
                    for(int k = 0; k < size_prev_layer; k++){
                        net_out[j] += weights[j][k] * net_in[k]; // weights * previous_layer_output
                    }
                }

                if(layer < len_dim - 1){
                    // ACTIVATION
                    for(int j = 0; j < this_block_size; j++){
                        if(net_out[j] < 0){
                            net_out[j] = 0; //relu
                        }
                    }
                    // FEED FORWARD 
                    for(int j = 0; j < num_proc_next_layer; j++){
                        MPI_Send(net_out, this_block_size, MPI_DOUBLE, j*len_dim + (layer+1), tag, MPI_COMM_WORLD);
                    }
                }
                else{
                    MPI_Send(net_out, this_block_size, MPI_DOUBLE, size, tag, MPI_COMM_WORLD);
                }
            }
            else{
                for(int j = 0; j < num_proc_prev_layer; j++){
                    MPI_Recv(&net_in[address_blocks_prev_layer[j]], size_blocks_prev_layer[j], MPI_DOUBLE, j*len_dim+(len_dim-1), tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last_size; j++){
                    res_exp[j] = exp(net_in[j]);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last_size; j++){
                    net_in[j] = res_exp[j]/sum;
                }
                ce -= log(net_in[y_train[i]]);
                if(largest(net_in, last_size) == y_train[i]){
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
                    for(int j = 0; j < num_proc_prev_layer; j++){
                        MPI_Recv(&net_in[address_blocks_prev_layer[j]], size_blocks_prev_layer[j], MPI_DOUBLE, j*len_dim+(layer-1), tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                double *net_out = (double*) malloc(this_block_size * sizeof(double));

                for(int j = 0; j < this_block_size; j++){
                    net_out[j] = weights[j][this_layer_size]; // bias
                    for(int k = 0; k < size_prev_layer; k++){
                        net_out[j] += weights[j][k] * net_in[k]; // weights * previous_layer_output
                    }
                }

                if(layer < len_dim - 1){
                    // ACTIVATION
                    for(int j = 0; j < this_block_size; j++){
                        if(net_out[j] < 0){
                            net_out[j] = 0; //relu
                        }
                    }
                    // FEED FORWARD 
                    for(int j = 0; j < num_proc_next_layer; j++){
                        MPI_Send(net_out, this_block_size, MPI_DOUBLE, j*len_dim + (layer+1), tag, MPI_COMM_WORLD);
                    }
                }
                else{
                    MPI_Send(net_out, this_block_size, MPI_DOUBLE, size, tag, MPI_COMM_WORLD);
                }
            }
            else{
                for(int j = 0; j < num_proc_prev_layer; j++){
                    MPI_Recv(&net_in[address_blocks_prev_layer[j]], size_blocks_prev_layer[j], MPI_DOUBLE, j*len_dim+(len_dim-1), tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last_size; j++){
                    res_exp[j] = exp(net_in[j]);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last_size; j++){
                    net_in[j] = res_exp[j]/sum;
                }
                b_ce -= log(net_in[y_test[i]]);
                if(largest(net_in, last_size) == y_test[i]){
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define size1 100
#define size2 101
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
    int last = dim[len_dim];
    if(rank != size){
        layer = rank/per_layer; // layer map
        layer_dim = dim[layer+1];
        prev_layer_dim = dim[layer];
        layer_part_size = layer_dim/per_layer;
        prev_layer_part_size = prev_layer_dim/per_layer;
        last = 0;
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
    double **B = (double**) malloc(layer_part_size * 2 * sizeof(double *)); // TODO << 1
    if(rank != size){
        for(int i = 0; i < layer_part_size * 2; i++){
            B[i] = (double*) malloc(dim[len_dim]) * sizeof(B));
        }
        for(int i = 0; i < layer_part_size * 2; i++){
            for(int j = 0; j < dim[len_dim]; j++){
                B[i][j] = (double(rand())/RAND_MAX - 0.5) / 4; // random
            }
        }
    }
    int **eye = (int**) malloc(last * sizeof(int *)); // TODO << 1
    if(rank == size){
        for(int i = 0; i < last; i++){
            eye[i] = (int*) malloc(last * sizeof(eye));
        }
        for(int i = 0; i < last * 2; i++){
            for(int j = 0; j < last; j++){
                if(i == j){
                    eye[i][j] = 1;
                }
                else{
                    eye[i][j] = 0;
                }
            }
        }
    }
    double *net_in = (double*) malloc(prev_layer_dim * sizeof(double));
    int *arr = (int*) malloc(last * sizeof(int));
    double *delta = (double*) malloc(dim[len_dim] * sizeof(double));
    double *res_exp = (double*) malloc(last * sizeof(double));
    double sum;
    
    /* TRAIN */
    double val_split = 0.2;
    int epochs = 10;
    double learn_rate = 0.05

    int t = 0, p = 0;
    int part = int(size1*(1-val_split));
    int tag = 0; // TODO image count?

    // TODO START COUNTER

    for(int epoch = 0; epoch < epochs; epoch++){
        for(int i = 0; i < part; i++){
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
            }
            else{
                for(int j = 0; j < per_layer; j++){
                    MPI_Recv(&net_in[j*dim[len_dim]/per_layer], dim[len_dim]/per_layer, MPI_DOUBLE, per_layer*(len_dim-1)+j, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                // SOFTMAX
                sum = 0;
                for(int j = 0; j < last; j++){
                    res_exp[j] = exp(net_in);
                    sum += res_exp[j];
                }
                for(int j = 0; j < last; j++){
                    net_in[j] = res_exp[j]/sum;
                    delta[j] = net_in[j] - eye[y_train[i]][j];
                }
            }
            delta = MPI_Bcast(delta, dim[len_dim], MPI_DOUBLE, size, MPI_COMM_WORLD)

        }
        
    }

    // TODO END COUNTER

    MPI_Finalize();
}
/*        
        for (x,y) in zip(input_train, output_train):
            if rank == size:
                res = []
                for i in range(per_layer):
                    res = np.append(res, comm.recv(source = (len_dim-1)*per_layer + i))
                res = np.exp(res)/np.sum(np.exp(res)) #softmax
                delta = res - outs_temp[y][0]
            delta = comm.bcast(delta, root = size)

            if rank != size:
                if layer < len_dim-1:
                    da = np.dot(B, delta)
                else:
                    delta_slice = delta[part*layer_part_size:(part+1)*layer_part_size]
                    da = np.append(delta_slice, delta_slice)

                dw = da[:layer_part_size]
                db = da[layer_part_size:]

                weights[:,:-1] += -learn_rate*np.tensordot(dw, net_in, axes=0)
                weights[:,-1] += -learn_rate*db

        tot = 0
        siz = len(input_test)
        ce = 0
        for (in_t,out_t) in zip(input_test,output_test):
            if rank != size:
                if layer == 0:
                    net_in = in_t
                
                else:
                    net_in = []
                    for i in range(per_layer):
                        net_in = np.append(net_in, comm.recv(source = (layer-1)*per_layer + i))

                
                net_out = np.zeros(layer_part_size)
                for i in range(layer_part_size):
                    net_out[i] = weights[i][-1] #bias
                    for j in range(dim[layer]):
                        net_out[i] += weights[i][j]*net_in[j] #weights * previous_layer_output

                if layer < len_dim-1:
                    net_out[net_out < 0] = 0 #relu

                if layer < len_dim-1:
                    for i in range(per_layer):
                        comm.send(net_out, dest = (layer+1)*per_layer + i)
                
                else:
                    comm.send(net_out, dest = size)

            if rank == size:
                res = []
                for i in range(per_layer):
                    res = np.append(res, comm.recv(source = (len_dim-1)*per_layer + i))
                res = np.exp(res)/np.sum(np.exp(res)) #softmax

                ce += -np.log(res[out_t[0]])
                if np.argmax(res) == out_t:
                    tot += 1

        b_tot = 0
        b_siz = len(blind_in)
        b_ce = 0
        for (in_t,out_t) in zip(blind_in,blind_out):
            if rank != size:
                if layer == 0:
                    net_in = in_t
                
                else:
                    net_in = []
                    for i in range(per_layer):
                        net_in = np.append(net_in, comm.recv(source = (layer-1)*per_layer + i))

                
                net_out = np.zeros(layer_part_size)
                for i in range(layer_part_size):
                    net_out[i] = weights[i][-1] #bias
                    for j in range(dim[layer]):
                        net_out[i] += weights[i][j]*net_in[j] #weights * previous_layer_output

                if layer < len_dim-1:
                    net_out[net_out < 0] = 0 #relu

                if layer < len_dim-1:
                    for i in range(per_layer):
                        comm.send(net_out, dest = (layer+1)*per_layer + i)
                
                else:
                    comm.send(net_out, dest = size)

            if rank == size:
                res = []
                for i in range(per_layer):
                    res = np.append(res, comm.recv(source = (len_dim-1)*per_layer + i))
                res = np.exp(res)/np.sum(np.exp(res)) #softmax
                    
                b_ce += -np.log(res[out_t[0]])
                if np.argmax(res) == out_t:
                    b_tot += 1

        if rank == size:
            print(f'Epoch {e+1}/{epochs}: Seen - Err: {ce/siz:.5f}; Acc: {tot/siz:.5f}. Unseen - Err: {b_ce/b_siz:.5f}; Acc: {b_tot/b_siz:.5f}.')
            #print(f'Total: {siz}; Total: {b_siz}.')
*/
/*
*/
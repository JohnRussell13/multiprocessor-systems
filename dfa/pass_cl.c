#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define size1 10
#define size2 10
#define in_len 14*14
#define len_dim 3

int main(){
    /* LOAD DATA */
    FILE *fp_xtr, *fp_ytr, *fp_xts, *fp_yts;
    double x_train[size1][in_len], y_train[size1], x_test[size2][in_len], y_test[size2];
    fp_xtr = fopen("img/x_train.txt", "r");
    fp_ytr = fopen("img/y_train.txt", "r");
    fp_xts = fopen("img/x_test.txt", "r");
    fp_yts = fopen("img/y_test.txt", "r");
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < in_len; j++){
            fscanf(fp_xtr, "%lf", x_train[i][j]);
            fscanf(fp_xts, "%lf", x_test[i][j]);
        }
        fscanf(fp_ytr, "%lf", y_train[i]);
        fscanf(fp_yts, "%lf", y_test[i]);
    }
    fclose(fp_xtr);
    fclose(fp_ytr);
    fclose(fp_xts);
    fclose(fp_yts);

    /* INIT */
    int dim[len_dim+1] = {in_len, 10, 10, 10};
    int dimVar[len_dim][2];
    for(int i = 0; i < len_dim; i++){
        dimVar[i][0] = dim[i+1];
        dimVar[i][1] = dim[i]+1;
    }
    double weights[len_dim]; //not a tensor but a list of matrices
}

// dimVar = []
// for i in range(len_dim):
//     dimVar.append((dim[i+1], (dim[i]+1)))
// weights = []
// for i in range(len_dim):
//     weights.append(np.random.normal(0, 0.1, dimVar[i]))

// test(input, weights, output, dim, 10)

// def net(input, weights, output_temp, len_dim, dim):
//     output = copy.deepcopy(output_temp)
//     output[0] = input
//     for i in range(len_dim):
//         for j in range(dim[i+1]):
//             output[i+1][j] += weights[i][j][-1]
//             for k in range(dim[i]):
//                 output[i+1][j] += weights[i][j][k]*output[i][k]
//         if (i+1) < len_dim:
//             output[i+1][output[i+1] < 0] = 0 #relu
//         else:
//           output[i+1] = np.exp(output[-1])/np.sum(np.exp(output[-1])) #softmax
//     return output

// def test(a, weights, b, dim, perc):
//     output_temp = []
//     for i in dim:
//         output_temp.append(np.zeros(i))
//     len_dim = len(dim)-1

//     per = int(len(a)*perc/100)
//     input = a[:per]
//     output = b[:per]

//     t = 0
//     p = 0

//     for i in range(per):
//         if i % (per/10) == 0:
//             print(f"Testing {p*10}%")
//             p += 1
//         if np.argmax(net(input[i], weights, output_temp, len_dim, dim)[-1]) == output[i][0]:
//             t += 1
//     print(f"Accuracy: {t}/{per}")
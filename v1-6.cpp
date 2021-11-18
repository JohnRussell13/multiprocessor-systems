#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc , char* argv []){
    int tc = strtol(argv[argc - 1], NULL , 10);
    int N;
    printf("Number: ");
    scanf("%d", &N);
    //int ind[N+1];
    int *ind = (int*) malloc((N+1) * sizeof(int));
    
    for(int i = 2; i <= N; i++){
        ind[i] = 0;
    }

    double s = omp_get_wtime ();

    for (int i = 2; i <= N; i++){
        if(ind[i] == 0){
            ind[i] = 1;
            #pragma omp parallel for num_threads(tc) schedule(static, 2)
            for(int j = i + i; j <= N; j = j + i){
                ind[j] = 2;
            }
        }
    }
    s = omp_get_wtime () - s;

    printf("Primes amongst the first %d integers are: ", N);
    for(int i = 0; i <= N; i++){
        if(ind[i] == 1){
            printf("%d ", i);
        }
    }
    printf("\n");
    printf("Static executed for %lf s\n", s);


    for(int i = 2; i <= N; i++){
        ind[i] = 0;
    }

    s = omp_get_wtime ();

    for (int i = 2; i <= N; i++){
        if(ind[i] == 0){
            ind[i] = 1;
            #pragma omp parallel for num_threads(tc) schedule(dynamic, 2)
            for(int j = i + i; j <= N; j = j + i){
                ind[j] = 2;
            }
        }
    }
    s = omp_get_wtime () - s;
    printf("Dynamic executed for %lf s\n", s);


    for(int i = 2; i <= N; i++){
        ind[i] = 0;
    }

    s = omp_get_wtime ();

    for (int i = 2; i <= N; i++){
        if(ind[i] == 0){
            ind[i] = 1;
            #pragma omp parallel for num_threads(tc) schedule(guided, 2)
            for(int j = i + i; j <= N; j = j + i){
                ind[j] = 2;
            }
        }
    }
    s = omp_get_wtime () - s;
    printf("Guided executed for %lf s\n", s);


    for(int i = 2; i <= N; i++){
        ind[i] = 0;
    }

    s = omp_get_wtime ();

    for (int i = 2; i <= N; i++){
        if(ind[i] == 0){
            ind[i] = 1;
            #pragma omp parallel for num_threads(tc) schedule(auto)
            for(int j = i + i; j <= N; j = j + i){
                ind[j] = 2;
            }
        }
    }
    s = omp_get_wtime () - s;
    printf("Auto executed for %lf s\n", s);


    for(int i = 2; i <= N; i++){
        ind[i] = 0;
    }

    s = omp_get_wtime ();

    for (int i = 2; i <= N; i++){
        if(ind[i] == 0){
            ind[i] = 1;
            #pragma omp parallel for num_threads(tc) schedule(runtime)
            for(int j = i + i; j <= N; j = j + i){
                ind[j] = 2;
            }
        }
    }
    s = omp_get_wtime () - s;
    printf("Runtime executed for %lf s\n", s);


    for(int i = 2; i <= N; i++){
        ind[i] = 0;
    }

    s = omp_get_wtime ();

    for (int i = 2; i <= N; i++){
        if(ind[i] == 0){
            ind[i] = 1;
            #pragma omp parallel for num_threads(tc)
            for(int j = i + i; j <= N; j = j + i){
                ind[j] = 2;
            }
        }
    }
    s = omp_get_wtime () - s;
    printf("N/A executed for %lf s\n", s);
    
    free(ind);
    ind = NULL;


    return 0;
}


    /*#pragma omp parallel num_threads (tc)
    {
        int trank = omp_get_thread_num();
        int low, high;

        if(trank != 0){
            low = trank * ceil(N/tc);
        }

        else{
            low = 2;
        }

        if(trank != tc - 1){
            high = low + ceil(N/tc);
        }
        else{
            high = N + 1;
        }

        while(1){

        }

        
        for(int i = low; i < high; i++){
            
        }
    }*/

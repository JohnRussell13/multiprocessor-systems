#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc , char* argv []){
    int tc = strtol(argv[argc - 1], NULL , 10);
    double N;
    double sum = 0;
    printf("Number: ");
    scanf("%lf", &N);
    double s = omp_get_wtime ();
    #pragma omp parallel num_threads (tc)
    {
        int trank = omp_get_thread_num();
        int low, high;

        low = trank * ceil(N/tc);

        if(trank != tc - 1){
            high = low + ceil(N/tc);
        }
        else{
            high = N + 1;
        }

        double part_sum = 0;
        for(int i = low; i < high; i++){
            part_sum += i;
        }
        //printf("Partial sum in thread %d is %d\n", trank, part_sum);
        #pragma omp barrier
        sum += part_sum;
    }
    s = omp_get_wtime () - s;
    printf("Total sum of the first %lf integers is %lf\n", N, sum);
    printf("Executed for %lf s\n", s);
    return 0;
}
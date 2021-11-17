#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc , char* argv []){
    int tc = strtol(argv[argc - 1], NULL , 10);
    double N;
    double sum = 0;
    int step; //range size (high-low)
    int part; //how many have step vs how many have step-1
    printf("Number: ");
    scanf("%lf", &N);
    step = ceil(N/tc);
    part = int(N) % tc;
    double s = omp_get_wtime ();
    #pragma omp parallel num_threads (tc) reduction(+: sum)
    {
	int trank = omp_get_thread_num();
    	if(trank < N){
		int low, high;

		if(trank < part){
			low = trank * step + 1;
			high = low + step;
		}
		else{
			low = part * step + (trank-part) * (step-1) + 1;
			high = low + step - 1;
		}

		/*if(trank != tc - 1){
		    high = low + ceil(N/tc);
		}
		else{
		    high = N + 1;
		}*/

		double part_sum = 0;
		for(int i = low; i < high; i++){
		    part_sum += i;
		}
		
		//printf("%d %d %d %lf\n", trank, low, high, part_sum);
		
		//printf("Partial sum in thread %d is %d\n", trank, part_sum);
		sum += part_sum;
        }
        #pragma omp barrier
    }
    s = omp_get_wtime () - s;
    printf("Total sum of the first %lf integers is %lf\n", N, sum);
    printf("Executed for %lf s\n", s);
    return 0;
}

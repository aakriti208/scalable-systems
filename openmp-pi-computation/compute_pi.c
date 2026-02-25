#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double step, pi;

int main(int argc, char *argv[]){
    if (argc != 2)
    {
        printf("Usage: %s <num_steps>\n", argv[0]);
        return 1;
    }

    int i, num_threads;
    double x, sum = 0.0;
    double start_time, end_time;
    long num_steps = atol(argv[1]);

    step = 1.0 / (double)num_steps;
    start_time = omp_get_wtime();

#pragma omp parallel for private(x) reduction(+ : sum)
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    end_time = omp_get_wtime();

#pragma omp parallel
{
    num_threads = omp_get_num_threads();
}

double execution_time = end_time - start_time;

printf("Pi = %f\n", pi);
printf("Number of threads used = %d\n", num_threads);
printf("Execution time = %f seconds \n", execution_time);

return 0;
}

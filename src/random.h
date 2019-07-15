// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#define Pi 3.14159265

double CauchyRand(double range){
	double u = 0.5;
	double cut = 10.0;
	srand(time(0));
	while(u == 0.5){
		u = rand();
	}

	u = range * tan(u*Pi);
	if(abs(u) > cut){
		return CauchyRand(range);
	}
	return u;
}

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

//include files to test
#include "euler.h"
#include "random.h"
/*
__global__ void CUDAHello (){
	printf("CUDA Hello\n");
}*/
void testRandom(){
	printf("%f\n", CauchyRand(0.3));
	printf("%f\n", CauchyRand(0.4));
	printf("%f\n", CauchyRand(0.5));
	printf("%f\n", CauchyRand(0.6));
	printf("%f\n", CauchyRand(0.7));
}

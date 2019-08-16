#ifndef SIGMOID_H_
#define SIGMOID_H_
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

double h_Logistic(double h_b, double h_t){
	double h_res  = (1/ (1 + exp(-(h_b*h_t))));
	if(h_res < 1.0)return 0;
	return h_res;
}

__device__ double Logistic(double b, double t){
	double res = (1/ (1 + exp(-(b*t))));
	if(res < 1.0)return 0;
	return res;
}

double Direction(double b){
//	printf("direction found: %f\n", b);
	if(b > 45.0 && b <= 135.0){
		return 0;
	}
	if(b > 135.0 && b <= 225.0){
		return 1;
	}
	if(b > 225.0 && b <= 315.0){
		return 2;
	}
	else{
//	if(b > 315.0 || b <= 45.0){
		return 3;
	}
	return -1;
}

#endif /* SIGMOID_H_ */

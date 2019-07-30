#ifndef SIGMOID_H_
#define SIGMOID_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

double Logistic(double b, double t){
	double res = (1/ (1 + exp(-(b*t))));
	if(res < 1.0)return 0;
	return res;
}

double Direction(double b){
	/*
	if(b > 0.0 && b <= 90.0){
		return 0;
	}
	if(b > 90.0 && b <= 180.0){
		return 1;
	}
	if(b > 180.0 && b <= 270.0){
		return 2;
	}
	else{
		return 3;
	}*/
//	b = (double)((int)b % 360);
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

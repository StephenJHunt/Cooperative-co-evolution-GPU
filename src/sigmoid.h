#ifndef SIGMOID_H_
#define SIGMOID_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

double Logistic(double b, double t){
	return (1/ (1 + exp(-(b*t))));
}

#endif /* SIGMOID_H_ */

#ifndef RANDOM_H_
#define RANDOM_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define Pi 3.14159265358979323846

double CauchyRand(double range){
	double u = 0.5;
	double cut = 10.0;
//	srand(time(0));
	while(u > 0.49 && u < 0.51){//reroll until a valid random
		u = 0+(1-0) * rand()/((double)RAND_MAX);
//		u = 0.49;
//		printf("%f\n", u);
	}

	u = range * tan(u*Pi);
//	printf("%f\n", u);
	if(abs(u) > cut){
		return CauchyRand(range);
	}
	return u;
}
#endif

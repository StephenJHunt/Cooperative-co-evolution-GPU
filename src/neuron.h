#ifndef NEURON_H_
#define NEURON_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

struct Neuron{
	bool Lesioned;
	int Trials;
	int Fitness;
	bool Tag;
	int Parent1;
	int Parent2;
	int ID;
	char* Name;
	double* Weight;
};

int counter = 0;
//srand




#endif

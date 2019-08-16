#ifndef NEURON_H_
#define NEURON_H_
#endif
#ifndef nPreds
#define nPreds = 3
#endif
#ifndef nHidden
#define nHidden = 15
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

//include other files
#include "random.h"

struct Neuron{
	int Parent1;
	int Parent2;
	int ID;
	char* Name;
//	double* Weight;
	double Weight[7];
	int size;
	bool Lesioned;
	int Trials;
	int Fitness;
	bool Tag;
};

int Ncounter = 0;

Neuron* newNeuron(int size){
	Ncounter++;
	double* w = new double[size];
	Neuron* n = new Neuron{-1, -1, Ncounter, "Neuron", {0, 0, 0, 0, 0, 0, 0}, size};

	return n;
}

Neuron createWeights(Neuron n, int size){
//	srand(time(0));
	for(int i=0;i<size;i++){
		double r = (double)((rand() % (13 - 6)) + 6);
		n.Weight[i] = r;
	}
	return n;
}

Neuron setFitness(Neuron n, int fitness){
	n.Fitness = n.Fitness + fitness;
	return n;
}

Neuron perturb(Neuron n, Neuron best, int size){
	if(!n.Tag){
		double coef = 0.3;
		for(int i=0;i<size;i++){
			n.Weight[i] = best.Weight[i] + CauchyRand(coef);
		}
	}
	//reset
	n.Fitness = 0;
	n.Trials = 0;
	return n;
}

void reset(Neuron* n){
	n->Fitness = 0;
	n->Trials = 0;
}




#endif

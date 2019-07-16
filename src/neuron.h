#ifndef NEURON_H_
#define NEURON_H_

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
	double* Weight;
	int size;
	bool Lesioned;
	int Trials;
	int Fitness;
	bool Tag;
};

int counter = 0;

Neuron* newNeuron(int size){
	counter++;
	double* w = new double[size];
	Neuron* n = new Neuron{-1, -1, counter, "Neuron", w, size};
	/*n.ID = counter;
	n.Weight = new double[size];
	n.Name = "Neuron";
	n.Parent1 = -1;
	n.Parent2 = -1;
	n.size = size;*/
	return n;
}

void createWeights(Neuron n, int size){
	srand(time(0));
	for(int i=0;i<size;i++){
		n.Weight[i] = (rand() * 12.0) - 6.0;
	}
}

void setFitness(Neuron n, int fitness){
	n.Fitness = n.Fitness + fitness;
}

void perturb(Neuron n, Neuron best, int size){
	if(!n.Tag){
		double coef = 0.3;
		for(int i=0;i<size;i++){
			n.Weight[i] = best.Weight[i] + CauchyRand(coef);
		}
	}
	//reset
	n.Fitness = 0;
	n.Trials = 0;
}

void reset(Neuron n){
	n.Fitness = 0;
	n.Trials = 0;
}




#endif

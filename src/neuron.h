#ifndef NEURON_H_
#define NEURON_H_
#endif
#ifndef nPreds
#define nPreds 6
#endif
#ifndef nHidden
#define nHidden 15
#endif
#ifndef weightSize
#define weightSize 7
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
	double Weight[weightSize];
	int size;
	bool Lesioned;
	int Trials;
	int Fitness;
	bool Tag;
};

int Ncounter = 0;

//Neuron* newNeuron(int size){
//	Ncounter++;
//	if(size > weightSize){
//		printf("ERROR IN newNeuron size param\n");
//	}
////	double* w = new double[size];
//	Neuron* n = new Neuron{-1, -1, Ncounter, "Neuron", {0, 0, 0, 0, 0, 0, 0}, size};
//
//	return n;
//}

Neuron newNeuron(int size, Neuron n){
	if(size > weightSize){
		printf("Weight size to large in function newNeuron");
		return n;
	}
	Ncounter++;
	n.Parent1 = -1;
	n.Parent2 = -1;
	n.ID = Ncounter;
	n.Name = "Neuron";
//	n.Weight = {0, 0, 0, 0, 0, 0, 0};
	n.size = size;
	n.Lesioned = false;
	n.Trials = 0;
	n.Fitness = 0;
	n.Tag = false;
	return n;
}

Neuron createWeights(Neuron n, int size){
//	srand(time(0));
	if(size > weightSize){
		printf("ERROR IN createWeights size param\n");
	}
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
	if(size > weightSize){
		printf("ERROR IN perturb size param\n");
	}
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

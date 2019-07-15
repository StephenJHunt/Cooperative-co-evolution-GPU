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
	double Weight[];
};

int counter = 0;
//srand

Neuron newNeuron(int size){
	counter++;
	Neuron n = new Neuron{};
	n.ID = counter;
	n.Weight = new double[size];
	n.Name = "Neuron";
	n.Parent1 = -1;
	n.Parent2 = -1;
	return n;
}

void createWeights(Neuron n, int size){
	srand(time(0));
	for(i=0;i<size;i++){
		n.Weight[i] = (rand() * 12.0) - 6.0;
	}
}

void setFitness(Neuron n, int fitness){
	n.Fitness = n.Fitness + fitness;
}

void perturb(Neuron n, Neuron best){

}






#endif

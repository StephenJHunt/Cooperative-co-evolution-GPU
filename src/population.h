#ifndef POPULATION_H_
#define POPULATION_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm>

//include other files
#include "random.h"
#include "neuron.h"

//struct Neurons{
//	Neuron Neuron[10];
//};
//
//Neurons newNeurons(int size){
//	Neurons n;
//	n.Neuron = new Neuron[size];
//	return n;
//}

struct Population{
	int ID;
	Neuron* Individuals;
	int numIndividuals;
	bool Evolvable;
	int NumToBreed;
	int GeneSize;
};

int popCounter = 0;

Population* newPopulation(int size, int genesize){
	popCounter++;
	Neuron* narr = new Neuron[size];
	Population* p = new Population{popCounter, narr, size, true, size/4, genesize};

	return p;
}
/*
bool Less(Neuron* n, int i, int j){
	Neuron arr = *n;
	Neuron n1 = arr[i];
	Neuron n2 = arr[j];
	int div1 = n1.Trials;
	int div2 = n2.Trials;
	if(div1 == 0){
		div1 = 1;
	}
	if(div2 == 0){
		div2 = 1;
	}
	return (n1.Fitness / div1) > (n2.Fitness / div2);
}*/

void createIndividuals(Population p){
	if(p.Evolvable){
		for(int i=0;i<p.numIndividuals;i++){
			Neuron n = newNeuron(p.GeneSize);
			createWeights(n, p.GeneSize);
			p.Individuals[i] = n;
		}
	}
}

Neuron selectNeuron(Population p){
	srand(time(0));
	int idx = rand() % p.numIndividuals;
//	printf("select neuron rand: %d\n", idx);
	return p.Individuals[idx];
}

bool operator<(Neuron n1, Neuron n2){
	int size = n1.size;
	int div1 = n1.Trials;
	int div2 = n2.Trials;
	if(div1 == 0){
		div1 = 1;
	}
	if(div2 == 0){
		div2 = 1;
	}
	return (n1.Fitness / div1) < (n2.Fitness / div2);
}

Population sortNeurons(Population p){
//	std::sort(p.Individuals, p.Individuals + p.numIndividuals);//is not working
	for(int i = 0;i<p.numIndividuals;i++){
		for(int j = 0;j<p.numIndividuals;j++){
			if(p.Individuals[i].Fitness > p.Individuals[j].Fitness){
				Neuron* temp = p.Individuals[i];
				p.Individuals[i] = p.Individuals[j];
				p.Individuals[j] = temp;
			}
		}

	}
	return p;

}

void onePointCrossover(Neuron* parent1, Neuron* parent2, Neuron* child1, Neuron* child2){
	srand(time(0));
	int crosspoint = rand() % parent1->size;
	for(int i = 0;i < parent1->size;i++){
		child1->Weight[i] = parent2->Weight[i];
		child2->Weight[i] = parent2->Weight[i];
	}
	child1->Parent1 = parent1->ID;
	child1->Parent2 = parent2->ID;
	child2->Parent1 = parent1->ID;
	child2->Parent2 = parent2->ID;

	reset(child1);
	reset(child2);

	for(int j=0;j<crosspoint;j++){
		double temp = child1->Weight[j];
		child1->Weight[j] = child2->Weight[j];
		child2->Weight[j] = temp;
	}
}

Population mate(Population p){
	srand(time(0));
	int mate;
	for(int i=0;i<p.NumToBreed;i++){
		if(i==0){
			mate = rand() % p.NumToBreed;
		}else{
			mate = rand() % i;
		}
		int childIndex1 = p.numIndividuals - (1 + (i *2));
		int childIndex2 = p.numIndividuals - (2 + (i *2));
		onePointCrossover(&p.Individuals[i], &p.Individuals[mate], &p.Individuals[childIndex1], &p.Individuals[childIndex2]);
	}
	return p;
}

Population mutate(Population p, double m){
	srand(time(0));
	for(int i=p.NumToBreed;i<p.numIndividuals;i++){
		if(((double)rand()) < m){
			int mutationIndex = rand() % p.GeneSize;
			p.Individuals[i].Weight[mutationIndex] = p.Individuals[i].Weight[mutationIndex] + CauchyRand(0.3);
		}
	}
	return p;
}

void growIndividuals(Population p){
	Neuron* arr = new Neuron[p.numIndividuals+1];
	double temp = 1.0;
	for(int i=0;i<p.numIndividuals;i++){
//		p.numIndividuals[i].Weight[i] = p.Individuals[i].Weight[i] + temp;
		//I don't actually know what this function is supposed to do
	}
}


#endif

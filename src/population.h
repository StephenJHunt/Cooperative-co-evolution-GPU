#ifndef POPULATION_H_
#define POPULATION_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

//include other files
#include "random.h"
#include "neuron.h"

struct Neurons{
	Neuron* Neurons;
};

struct Population{
	int ID;
	Neurons* Individuals;
	int numIndividuals;
	bool Evolvable;
	int NumToBreed;
	int GeneSize;
};

bool isLess(Neurons n, int i, int j){
	Neuron n1 = n.Neurons[i];
	Neuron n2 = n.Neurons[j];
	int div1 = n1.Trials;
	int div2 = n2.Trials;
	if(div1 == 0){
		div1 = 1;
	}
	if(div2 == 0){
		div2 = 1;
	}
	return (n1.Fitness / div1) > (n2.Fitness / div2);
}

int popCounter = 0;

Population* newPopulation(int size, int genesize){
	popCounter++;
	Neuron* ns = new Neuron[size];
	Neurons* indivs = new Neurons{ns};
	Population* p = new Population{counter, indivs, size, true, size/4, genesize};
	/*p.ID = counter;
	p.numIndividuals = size;
	p.Individuals = new Neurons[size];
	p.Evolvable = true;
	p.GeneSize = genesize;
	p.NumToBreed = size/4;*/
	return p;
}

void createIndividuals(Population p){
	if(p.Evolvable){
		for(int i=0;i<p.numIndividuals;i++){
			Neuron* n = newNeuron(p.GeneSize);
			p.Individuals[i] = n;
			p.Individuals[i].createWeights(p.Individuals[i]. p.GeneSize);
		}
	}
}

Neuron selectNeuron(Population p){
	srand(time(0));
	int idx = rand() % p.numIndividuals;
	return p.Individuals[idx];
}

void sortNeurons(Population p){
	sort(p.Individuals);
}

void onePointCrossover(Neuron parent1, Neuron parent2, Neuron child1, Neuron child2){
	srand(time(0));
	int crosspoint = rand() % parent1.size;
	for(int i = 0;i < parent1.size;i++){
		child1.Weight[i] = parent2.Weight[i];
		child2.Weight[i] = parent2.Weight[i];
	}
	child1.Parent1 = parent1.ID;
	child1.Parent2 = parent2.ID;
	child2.Parent1 = parent1.ID;
	child2.Parent2 = parent2.ID;

	reset(child1);
	reset(child2);

	for(int j=0;j<crosspoint;j++){
		double temp = child1.Weight[j];
		child1.Weight[j] = child2.Weight[j];
		child2.Weight[j] = temp;
	}
}

void mate(Population p){
	srand(time(0));
	int mate;
	for(int i=0;i<p.NumToBreed;i++){
		if(i==0){
			mate = rand() % p.NumToBreed;
		}else{
			mate = rand() % i;
		}
		int childIndex1 = p.NumIndividuals - (1 + (i *2));
		int childIndex2 = p.NumIndividuals - (2 + (i *2));
		onePointCrossover(p.Individuals[i], p.Individuals[mate], p.Individuals[childIndex1], p.Individuals[childIndex2]);
	}
}

void Mutate(Population p, double m){
	srand(time(0));
	for(int i=p.NumToBreed;i<p.numIndividuals;i++){
		if(((double)rand()) < m){
			int mutationIndex = rand() % p.GeneSize;
			p.Individuals[i].Weight[mutationIndex] = p.Individuals[i].Weight[mutationIndex] + CauchyRand(0.3);
		}
	}
}

void growIndividuals(Population p){
	for(i=0;i<p.numIndividuals;i++){
		double temp = 1.0;
		//need to somehow add this temp value to the end of the weight array
	}
}


#endif

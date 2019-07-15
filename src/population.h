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
	Neuron Neurons[];
};

struct Population{
	int ID;
	Neurons Individuals;
	int numIndividuals;
	bool Evolvable;
	int NumToBreed;
	int GeneSize;
};

bool isLess(Neurons n, int i, int j){
	int div1 = n.Neurons[i].Trials;
	int div2 = n.Neurons[j].Trials;
	if(div1 == 0){
		div1 = 1;
	}
	if(div2 == 0){
		div2 = 1;
	}
	return (n.Neurons[i].Fitness / div1) > (n.Neurons[j].Fitness / div2);
}

int counter = 0;

Population newPopulation(int size, int genesize){
	counter++;
	Population p = new Population{};
	p.ID = counter;
	p.numIndividuals = size;
	p.Individuals = new Neurons[size];
	p.Evolvable = true;
	p.GeneSize = genesize;
	p.NumToBreed = size/4;
}

void createIndividuals(Population p){
	if(p.Evolvable){
		for(i=0;i<p.numIndividuals;i++){
			p.Individuals[i] = newNeuron(p.GeneSize);
			p.Individuals[i].createWeights(p.Individuals[i]. p.Genesize);
		}
	}
}

Neuron selectNeuron(Population p){
	srand(time(0));
	idx = rand() % p.numIndividuals;
	return p.Individuals[idx];
}

void sortNeurons(Population p){
	sort(p.Individuals);
}

void mate(Population p){
	srand(time(0))
	int mate;
	for(i=0;i<p.NumToBreed;i++){
		if(i==0){
			mate = rand() % p.NumToBreed;
		}else{
			mate = rand() % i;
		}
		int childIndex1 = p.NumIndividuals - (1 + (i *2));
		int childIndex2 = p.NumIndividuals - (2 + (i *2));
		//one point crossover
	}
}


#endif

#ifndef POPULATION_H_
#define POPULATION_H_
#ifndef nPreds
#define nPreds 6
#endif
#ifndef nHidden
#define nHidden 50
#endif
#ifndef nIndivs
#define nIndivs 540
#endif
#ifndef weightSize
#define weightSize 7
#endif
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
//	Neuron* Individuals;
	Neuron Individuals[nIndivs];
	int numIndividuals;
	bool Evolvable;
	int NumToBreed;
	int GeneSize;
};

int popCounter = 0;

Population* newPopulation(int size, int genesize){
	popCounter++;
	if(size > nIndivs){
		printf("ERROR IN newPopulation size param\n");
	}
//	Neuron* narr = new Neuron[size];
//	Neuron* n = new Neuron;
	Population* p = new Population;//{popCounter, size, true, size/4, genesize};
//	for(int i =0;i<size;i++){
//		p->Individuals[i] = *n;
//	}
	p->Evolvable = true;
	p->GeneSize=genesize;
	p->ID=popCounter;
	p->NumToBreed=size/4;
	p->numIndividuals=size;
	return p;
}

void createIndividuals(Population* p){
	if(p->Evolvable){
		for(int i=0;i<p->numIndividuals;i++){
			Neuron n = newNeuron(p->GeneSize, p->Individuals[i]);
			n = createWeights(n, p->GeneSize);
			p->Individuals[i] = n;
		}
	}
}

Neuron selectNeuron(Population p){
	srand(time(0));
	int idx = rand() % p.numIndividuals;
	if(idx > nIndivs || idx < 0){
		printf("Index out of range in function selectNeuron\n");
	}
//	printf("select neuron rand: %d\n", idx);
	return p.Individuals[idx];
}

Population sortNeurons(Population p){
//	std::sort(p.Individuals, p.Individuals + p.numIndividuals);//is not working
	for(int i = 0;i<p.numIndividuals;i++){
		for(int j = 0;j<p.numIndividuals;j++){
			if(p.Individuals[i].Fitness > p.Individuals[j].Fitness){
				Neuron temp = p.Individuals[i];
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

	if(parent1->size != child1->size || parent1->size != child2->size || parent2->size != child1->size || parent2->size != child2->size){
		printf("Neuron size mismatches in function onePointCrossover\n");
		return;
	}

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

	if(crosspoint > weightSize){
		printf("Crosspoint index out of bounds in function onePointCrossover\n");
		return;
	}

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
		if(mate > nIndivs || mate <0){
			printf("Mate index 1 out of bounds in function mate\n");
			break;
		}
		int childIndex1 = p.numIndividuals - (1 + (i *2));
		if(childIndex1 > nIndivs || childIndex1 <0){
			printf("Child index 1 out of bounds in function mate\n");
			break;
		}
		int childIndex2 = p.numIndividuals - (2 + (i *2));
		if(childIndex2 > nIndivs || childIndex2 <0){
			printf("Child index 2 out of bounds in function mate\n");
			break;
		}
		onePointCrossover(&p.Individuals[i], &p.Individuals[mate], &p.Individuals[childIndex1], &p.Individuals[childIndex2]);
	}
	return p;
}

Population mutate(Population p, double m){
	srand(time(0));
	for(int i=p.NumToBreed;i<p.numIndividuals;i++){
		if(((double)rand()) < m){
			int mutationIndex = rand() % p.GeneSize;
			if(mutationIndex > weightSize || weightSize < 0){
				printf("Mutation Index out of bounds in function mutate\n");
				break;
			}
			p.Individuals[i].Weight[mutationIndex] = p.Individuals[i].Weight[mutationIndex] + CauchyRand(0.3);
		}
	}
	return p;
}



#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <random>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

//include files to test
#include "euler.h"
#include "random.h"
#include "neuron.h"
#include "population.h"
#include "sigmoid.h"
#include "feedforward.h"
#include "network.h"
#include "environment.h"
#include "predatorprey.h"
/*
__global__ void CUDAHello (){
	printf("CUDA Hello\n");
}*/
void testRandom(){
	printf("%f\n", CauchyRand(0.3));
	printf("%f\n", CauchyRand(0.4));
	printf("%f\n", CauchyRand(0.5));
	printf("%f\n", CauchyRand(0.6));
	printf("%f\n", CauchyRand(0.7));
}
void testSigmoid(){
	printf("\n%f\n", Logistic(1.0, 2.0));
}
void testPop(){
	Population* p = newPopulation(10, 10);
//	printf("am here\n");
	Population deref = *p;
//	Neuron nr = deref.Individuals[0];

	createIndividuals(deref);
	Neuron t = selectNeuron(deref);
	mate(deref);
	mutate(deref, 0.3);
//	growIndividuals(deref);

//	Neuron* testn = newNeuron(deref.GeneSize);
//	printf("%f\n", testn->Weight[0]);
//	createWeights(*testn, deref.GeneSize);
//	deref.Individuals[0] = *testn;
//	nr = deref.Individuals[0];
//	createIndividuals(deref);
//	Neurons ns = deref.Individuals;
//	Neuron* n = ns.Neuron;
//	Neuron nr0 = deref.Individuals[1];
//	Neuron nr4 = deref.Individuals[3];
//	Neuron nr1 = deref.Individuals[7];
//	Neuron nr2 = deref.Individuals[8];
//	Neuron nr3 = deref.Individuals[9];
//	printf("%d\n%d\n%d\n%d\n%d\n%d\n%d",deref.NumToBreed, nr.Fitness, nr0.Fitness, nr4.Fitness, nr1.Fitness, nr2.Fitness, nr3.Fitness);
//	printf("%f\n", nr.Weight[0]);
	printf("Population Tests Passed\n");
}

void testFF(){
	feedForward* fdfw = newFeedForward(5, 5, 5, true);
}

class useNetwork: public Network{
	public:
		bool hasBias(){
			return true;
		}
};

void testNetwork(){
	useNetwork network;
	printf("Network test ");
	printf(network.hasBias() ? "true\n":"false\n");
}

class useEnvironment: public Environment{
public:
	bool Caught(){
		return true;
	}
};

void testEnvironment(){
	useEnvironment environment;
	printf("Environment test ");
	printf(environment.Caught() ? "true\n":"false\n");
}

void testPredatorPrey(){
	PredatorPrey* pp = newPredatorPrey(5);
	reset(*pp, 1);
}

void runTests(){
	testRandom();
//	testSigmoid();
//	testPop();
//	testFF();
//	testNetwork();
//	testEnvironment();
//	testPredatorPrey();

}


////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

//include other files
#include "test.h"
#include "random.h"
//#include "environment.h"
#include "predatorprey.h"
//#include "network.h"
#include "feedforward.h"
#include "population.h"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

//globals
feedForward* bestTeam;
Population* subPops;
Population** predSubPops;//Population** ?
Gridworld world;
int catches;

//input params with default values
bool sim = false;
int hidden = 10;
int numIndivs = 100;
int numInputs = 2;
int numOutputs = 5;
int burstGens = 10;
int maxGens = 100000;
int goalFitness = 100000;
int numPreds = 3;
int trialsPerEval = 9;

struct tempState{
	int* PredatorX;
	int* PredatorY;
	int PreyX;
	int PreyY;
};

Population* init(int hid, int num, int genes){
	Population* pops = new Population[hid];
	for(int i = 0; i < hid; i++){
		Population* p = newPopulation(num, genes);
		createIndividuals(*p);
		pops[i] = *p;
	}
	return pops;
}

int calculateDistance(int predX, int predY, int preyX ,int preyY){
	double xDist = 0;
	double yDist = 0;

	xDist = abs((double)(predX-preyX));
	if(xDist > double(world.length/2)){
		xDist = double(world.length) - xDist;
	}

	yDist = abs((double)(predY-preyY));
	if(yDist > double(world.height/2)){
		yDist = double(world.height) - yDist;
	}
	return int(xDist + yDist);
}

feedForward* evaluate(PredatorPrey e, feedForward* team, int numTeams){
	catches = 0;
	int total_fitness = 0;

	int PreyPositions[2][9] = {{16, 50, 82, 82, 82, 16, 50, 50, 82},{50, 50, 50, 82, 16, 50, 16, 82, 50}};

	for(int l = 0;l < trialsPerEval;l++){
		int fitness =0;
		int steps = 0;
		int maxSteps = 150;
		int avg_init_dist = 0;
		int avg_final_dist = 0;

		int inplen = getTotalInputs(team[0]);
		int outlen = getTotalOutputs(team[0]);
		double* input = new double[inplen];
		double* output = new double[outlen];
		State state;

		setPreyPosition(e, *PreyPositions[l], *PreyPositions[l]);
		State* statepntr = getState(e);
		Gridworld* worldpntr = getWorld(e);
		state = *statepntr;
		world = *worldpntr;

		int nearestDist = 0;
		int nearestPred = 0;
		int currentDist = 0;

		for(int p = 0 ; p < numPreds; p++){
			avg_init_dist = avg_init_dist + calculateDistance(state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
		}
		avg_init_dist = avg_init_dist/numPreds;

		while(!Caught(e) && steps < maxSteps){
			for(int p=0; p < numPreds;p++){
				currentDist = calculateDistance(state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
				if(currentDist<nearestDist){
					nearestDist = currentDist;
					nearestPred = p;
				}
			}

			PerformPreyAction(e, nearestPred);

			for(int pred = 0; pred < numTeams;pred++){
				input[0] = double(e.state->PreyX);
				input[1] = double(e.state->PreyY);
				delete[] output;
				output = new double[outlen];//reset output in between?
				double* out = Activate(team[pred], input, inplen, output);
				PerformPredatorAction(e, pred, out, team[pred].NumOutputs);
//				printf("\n");
			}
			State* ts = getState(e);
			state = *ts;
			steps++;
/*
			//output state
			for(int pred = 0;pred < numPreds;pred++){
				printf("Predator %d, %d\n", state.PredatorX[pred], state.PredatorY[pred]);
			}
			printf("prey %d, %d \n", state.PreyX, state.PreyY);
*/
		}

		if(Caught(e)){
			if(sim == true){
				printf("Simulation Complete\n");
				printf("Predator at position %d, %d caught the prey at position %d, %d after %d steps", state.PredatorX[nearestPred], state.PredatorY[nearestPred], state.PreyX, state.PreyY, steps);
			}
		}

		for(int p = 0; p < numPreds;p++){
//			printf("Predator %d distance: %d\n", p, calculateDistance(state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY));
			avg_final_dist = avg_final_dist + calculateDistance(state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
//			printf("avg final dist: %d\n", avg_final_dist);
		}
		avg_final_dist = avg_final_dist/numPreds;

		if(!Caught(e)){
//			printf("fitness eval: avg init:%d, avg final:%d, result:%d\n", avg_init_dist, avg_final_dist, ((avg_init_dist - avg_final_dist)));
			fitness = (avg_init_dist - avg_final_dist);// /10
		}else{
			fitness = (200 - avg_final_dist)/10;
			catches++;
		}
		total_fitness = total_fitness + fitness;
	}

	for(int pred = 0; pred < numTeams;pred++){
		setFitness(team[pred], (total_fitness/trialsPerEval));
		setCatches(team[pred], catches);
		setNeuronFitness(team[pred]);
	}
	printf("fitness before return %d\n", total_fitness);
	return team;

}


int main(int argc, char **argv)
{
//	runTests();
//	printf("\n\nExecution\n\n");


	numInputs = 2;
	hidden = 15;
	numOutputs = 5;
	numIndivs = 540;//540
	maxGens = 100;
	goalFitness = 100;
	numPreds = 3;//6
	burstGens = 2;


	//parse input

	bool stagnated;
	double mutationRate = 0.4;
//	printf("Number of inputs is %d\n", numInputs);
//	printf("Number of hidden units is %d\n", hidden);
//	printf("Number of outputs is %d\n", numOutputs);
//	printf("Number of individuals per population is %d\n", numIndivs);
//	printf("Max number of generations is %d\n", maxGens);
//	printf("Mutation rate is %f\n", mutationRate);
//	printf("Burst mutate after %d generations\n", burstGens);
//	printf("Number of predators is %d\n", numPreds);
//  printf("Number of team trials per evaluation is %d\n", trialsPerEval);

//	int* performanceQueue = new int[burstGens];
	int bestFitness = 0;
	int generations = 0;
	stagnated = false;

	predSubPops = new Population*[numPreds];
	//initialisation
	for(int p = 0;p<numPreds;p++){
		feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
		Population* subpops = init(hidden, numIndivs, ff->GeneSize);
		predSubPops[p] = subpops;
	}

	int numTrials = 10 * numIndivs;
	feedForward* team = new feedForward[numPreds];
	while(generations < maxGens && catches < numTrials){
		catches = 0;

		for(int x = 0; x < numTrials;x++){

			for(int f = 0;f<numPreds;f++){
				feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
				Create(*ff, predSubPops[f], hidden);
				team[f] = *ff;
			}
			PredatorPrey* pp = newPredatorPrey(numPreds);
			reset(*pp, numPreds);

			feedForward* t = evaluate(*pp, team, numPreds);
			catches = catches + getCatches(t[0]);
			if(bestFitness == 0 && !teamfound){
				bestFitness = getFitness(t[0]);
			}
			printf("best fitness %d\n", bestFitness);
			printf("this team fitness: %d\n", getFitness(t[0]));
			if(getFitness(t[0]) > bestFitness){
				bestFitness = getFitness(t[0]);
				double* bestActivation = new double[t->numHidden];
				Neuron* bestNeurons = new Neuron[t->numHidden];
				for(int i = 0;i<t->numHidden;i++){
					bestActivation[i] = t->Activation[i];
					bestNeurons[i] = t->HiddenUnits[i];
				}
				bestTeam = new feedForward;
				bestTeam->ID = t->ID;
				bestTeam->Catches = t->Catches;
				bestTeam->Fitness = t->Fitness;
				bestTeam->GeneSize = t->GeneSize;
				bestTeam->NumInputs = t->NumInputs;
				bestTeam->NumOutputs = t->NumOutputs;
				bestTeam->Parent1 = t->Parent1;
				bestTeam->Parent2 = t->Parent2;
				bestTeam->Trials = t->Trials;
				bestTeam->bias = t->bias;
				bestTeam->name = t->name;
				bestTeam->numHidden = t->numHidden;
				bestTeam->Activation=bestActivation;
				bestTeam->HiddenUnits = bestNeurons;
				for(int i = 0;i<numPreds;i++){
					Tag(bestTeam[i]);
				}
			}
			if(!teamfound){
				teamfound = true;
				double* bestActivation = new double[t->numHidden];
				Neuron* bestNeurons = new Neuron[t->numHidden];
				for(int i = 0;i<t->numHidden;i++){
					bestActivation[i] = t->Activation[i];
					bestNeurons[i] = t->HiddenUnits[i];
				}
				bestTeam = new feedForward;
				bestTeam->ID = t->ID;
				bestTeam->Catches = t->Catches;
				bestTeam->Fitness = t->Fitness;
				bestTeam->GeneSize = t->GeneSize;
				bestTeam->NumInputs = t->NumInputs;
				bestTeam->NumOutputs = t->NumOutputs;
				bestTeam->Parent1 = t->Parent1;
				bestTeam->Parent2 = t->Parent2;
				bestTeam->Trials = t->Trials;
				bestTeam->bias = t->bias;
				bestTeam->name = t->name;
				bestTeam->numHidden = t->numHidden;
				bestTeam->Activation=bestActivation;
				bestTeam->HiddenUnits = bestNeurons;
			}
		}

		printf("Generation %d, best fitness is %d, catches is %d\n", generations, bestFitness, catches);

		if(generations%burstGens == 0 && generations != 0){
			//burst mutate
			stagnated = true;

			for(int pred = 0; pred < numPreds; pred++){
				Population* predPop = predSubPops[pred];
				for(int i = 0; i< numIndivs;i++){
					Population subpop = predPop[i];
					for(int n = 0; n< numIndivs;n++){
						Neuron indiv = subpop.Individuals[i];
						Neuron* hid = getHiddenUnits(bestTeam[0]);
						perturb(indiv, hid[i], numIndivs);
					}
				}
			}
		}
		if(!stagnated){
			for(int i = 0 ;i<numPreds;i++){
				for(int j = 0;j<hidden;j++){
					sortNeurons(predSubPops[i][j]);
					mate(predSubPops[i][j]);
					mutate(predSubPops[i][j], mutationRate);
				}
			}
		}
		stagnated = false;
		//reset teams?? doens't seem to do anything
		generations++;
	}
//	char* test = "hello";//this is a string now
	//printf(test);
	//printf("\n");
    //printf("Hello World!\n");
    //CUDAHello<<<1,10>>>();
//    cudaDeviceReset();
    //
}


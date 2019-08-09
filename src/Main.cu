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

__device__ int calculateDistance(Gridworld* world, int predX, int predY, int preyX ,int preyY){
	double xDist = 0;
	double yDist = 0;

	xDist = abs((double)(predX-preyX));
	if(xDist > double(world->length/2)){
		xDist = double(world->length) - xDist;
	}

	yDist = abs((double)(predY-preyY));
	if(yDist > double(world->height/2)){
		yDist = double(world->height) - yDist;
	}
	return int(xDist + yDist);
}

__global__ void kernelAssignFitness(int fitness, Neuron** hiddenUnits){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

//    for(int i=index;i<numHidden;i++){
//		Neuron* n = hiddenUnits[i];
		hiddenUnits[index]->Fitness = fitness;
		hiddenUnits[index]->Trials++;
//		hiddenUnits[i] = n;
//    }
}

__global__ void kernelOnePreyMovement(PredatorPrey pp, int nearest){

}
/*
feedForward* evaluate(PredatorPrey e, feedForward* team, int numTeams){
	catches = 0;
	int total_fitness = 0;
	int SMs;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, deviceID);
	int threadsPerBlock = 256;
	int blocks = 32 * SMs;

	int PreyPositions[2][9] = {{16, 50, 82, 82, 82, 16, 50, 50, 82},{50, 50, 50, 82, 16, 50, 16, 82, 50}};

	for(int l = 0;l < trialsPerEval;l++){//parallel?
		int fitness =0;
		int steps = 0;
		int maxSteps = 150;
		int avg_init_dist = 0;
		int avg_final_dist = 0;

		//do these before with cudaMallocManaged
		int inplen = getTotalInputs(team[0]);
		int outlen = getTotalOutputs(team[0]);
		double* input = new double[inplen];
		double* output = new double[outlen];
		State state;

		setPreyPosition(e, PreyPositions[0][l], PreyPositions[1][l]);
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

		while(!Caught(e) && steps < maxSteps){//paralellise so that always runs maxSteps?
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
//			delete[] input;
//			delete[] output;
///*
			//output state
			for(int pred = 0;pred < numPreds;pred++){
				printf("Predator %d, %d\n", state.PredatorX[pred], state.PredatorY[pred]);
			}
			printf("prey %d, %d \n", state.PreyX, state.PreyY);
//

		}

		if(Caught(e)){
			if(sim == true){
				printf("Simulation Complete\n");
				printf("Predator at position %d, %d caught the prey at position %d, %d after %d steps", state.PredatorX[nearestPred], state.PredatorY[nearestPred], state.PreyX, state.PreyY, steps);
			}
		}

		for(int p = 0; p < numPreds;p++){
			avg_final_dist = avg_final_dist + calculateDistance(state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
		}
		avg_final_dist = avg_final_dist/numPreds;

		if(!Caught(e)){
			fitness = (avg_init_dist - avg_final_dist);// /10
		}else{
			fitness = (200 - avg_final_dist)/10;
			catches++;
		}
		total_fitness = total_fitness + fitness;
	}

	for(int pred = 0; pred < numTeams;pred++){
		team[pred].Fitness = (total_fitness); // /trialsPerEval
		team[pred].Catches = catches;
		Neuron** d_neurons;
		// <<<blocks, threadsPerBlock>>>
		int numBytes = team[pred].numHidden * sizeof(team[pred].HiddenUnits[0]);
		//case 1
//		cudaMalloc(&d_neurons, numBytes);//optimise to only take neuron fitness and trials not whole struct
//		cudaMemcpy(team[pred].HiddenUnits, d_neurons, numBytes, cudaMemcpyHostToDevice);
//		kernelAssignFitness<<<1, team[pred].numHidden>>>(total_fitness, d_neurons);
//		cudaDeviceSynchronize();
//		cudaMemcpy(team[pred].HiddenUnits, d_neurons, numBytes, cudaMemcpyDeviceToHost);
		//case 2
		kernelAssignFitness<<<1, team[pred].numHidden>>>(total_fitness, team[pred].HiddenUnits);
		cudaDeviceSynchronize();
		for(int i = 0; i<team[pred].numHidden;i++){
			Neuron* n = team[pred].HiddenUnits[i];
			n->Fitness = team[pred].Fitness;
			n->Trials++;
			team[pred].HiddenUnits[i] = n;
		}
	}
	return team;

}
*/

__global__ void runEvaluationsParallel(State* statepntr, Gridworld* worldpntr, feedForward** teams, int numPreds, double* input, double* output, int inplen, int outlen, int trialsPerEval, bool sim, int numTrials){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index;i < numTrials;i+= stride){
		int catches = 0;
		int total_fitness = 0;

		int PreyPositions[2][9] = {{16, 50, 82, 82, 82, 16, 50, 50, 82},{50, 50, 50, 82, 16, 50, 16, 82, 50}};

		for(int l = 0;l < trialsPerEval;l++){//parallel?
			int fitness =0;
			int steps = 0;
			int maxSteps = 150;
			int avg_init_dist = 0;
			int avg_final_dist = 0;

			State state;
			Gridworld world;

			setPreyPosition(statepntr, PreyPositions[0][l], PreyPositions[1][l]);//use state instead of PredatorPrey?
//			State* statepntr = e->state;
//			Gridworld* worldpntr = e->world;
			state = *statepntr;
			world = *worldpntr;

			int nearestDist = 0;
			int nearestPred = 0;
			int currentDist = 0;

			for(int p = 0 ; p < numPreds; p++){
				avg_init_dist = avg_init_dist + calculateDistance(worldpntr, state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
			}
			avg_init_dist = avg_init_dist/numPreds;

			while(!Caught(statepntr) && steps < maxSteps){//paralellise so that always runs maxSteps?
				for(int p=0; p < numPreds;p++){
					currentDist = calculateDistance(worldpntr, state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
					if(currentDist<nearestDist){
						nearestDist = currentDist;
						nearestPred = p;
					}
				}

				PerformPreyAction(statepntr, worldpntr, nearestPred);

				for(int pred = 0; pred < numPreds;pred++){
					input[0] = double(statepntr->PreyX);
					input[1] = double(statepntr->PreyY);
					delete[] output;
					output = new double[outlen];
					double* out = Activate(teams[i][pred], input, inplen, output);
					PerformPredatorAction(statepntr, worldpntr, pred, out, teams[i][pred].NumOutputs);//change to use state?
				}
				steps++;
			}
			if(Caught(statepntr)){
				if(sim == true){
					printf("Simulation Complete\n");
					printf("Predator at position %d, %d caught the prey at position %d, %d after %d steps", state.PredatorX[nearestPred], state.PredatorY[nearestPred], state.PreyX, state.PreyY, steps);
				}
			}

			for(int p = 0; p < numPreds;p++){
				avg_final_dist = avg_final_dist + calculateDistance(worldpntr, state.PredatorX[p], state.PredatorY[p], state.PreyX, state.PreyY);
			}
			avg_final_dist = avg_final_dist/numPreds;

			if(!Caught(statepntr)){
				fitness = (avg_init_dist - avg_final_dist);// /10
			}else{
				fitness = (200 - avg_final_dist)/10;
				catches++;
			}
			total_fitness = total_fitness + fitness;
		}

		for(int pred = 0; pred < numPreds;pred++){
			teams[i][pred].Fitness = (total_fitness); // /trialsPerEval
			teams[i][pred].Catches = catches;
			Neuron** d_neurons;
			// <<<blocks, threadsPerBlock>>>
			int numBytes = teams[i][pred].numHidden * sizeof(teams[i][pred].HiddenUnits[0]);
			//case 1
	//		cudaMalloc(&d_neurons, numBytes);//optimise to only take neuron fitness and trials not whole struct
	//		cudaMemcpy(team[pred].HiddenUnits, d_neurons, numBytes, cudaMemcpyHostToDevice);
	//		kernelAssignFitness<<<1, team[pred].numHidden>>>(total_fitness, d_neurons);
	//		cudaDeviceSynchronize();
	//		cudaMemcpy(team[pred].HiddenUnits, d_neurons, numBytes, cudaMemcpyDeviceToHost);
			//case 2
	//		kernelAssignFitness<<<1, team[pred].numHidden>>>(total_fitness, team[pred].HiddenUnits);
	//		cudaDeviceSynchronize();
			for(int i = 0; i<teams[i][pred].numHidden;i++){
				Neuron* n = teams[i][pred].HiddenUnits[i];
				n->Fitness = teams[i][pred].Fitness;
				n->Trials++;
				teams[i][pred].HiddenUnits[i] = n;
			}
		}
//		printf("Team %d's fitness: %d\n",i , teams[i][0].Fitness);
	}
//	return team;
}

void CHECK(cudaError_t err){
	if(err){
		printf("Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
}

int main(int argc, char **argv)
{
	//testing values
	numInputs = 2;
	hidden = 15;
	numOutputs = 5;
	numIndivs = 100;//540
	maxGens = 100;
	goalFitness = 100;
	numPreds = 3;//6
	burstGens = 2;


	//TODO: parse input

	//simulation values
	bool stagnated;
	double mutationRate = 0.4;
	int bestFitness = 0;
	int generations = 0;
	stagnated = false;
	bool teamfound = false;
	int numTrials = 10 * numIndivs;

	//GPU values
	int SMs;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, deviceID);
	int threadsPerBlock = 256;
	int blocks = 32 * SMs;

	predSubPops = new Population*[numPreds];
	//initialisation of subpopulations
	for(int p = 0;p<numPreds;p++){
		feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
		Population* subpops = init(hidden, numIndivs, ff->GeneSize);
		predSubPops[p] = subpops;
	}

	feedForward teams[numTrials][numPreds];
	feedForward** d_teams;
	feedForward* h_team;
	feedForward* d_team;
	int numBytes = numTrials * (numPreds * sizeof(feedForward));
//	cudaMallocManaged(&teams, numBytes);
	CHECK(cudaMalloc((void **)&d_teams, numBytes));
	for (int t =0;t < numTrials;t++){
		numBytes = numPreds * sizeof(feedForward);
		cudaMalloc(&d_team, numBytes);
//		cudaMallocManaged(&team, numBytes);
		teams[t] = h_team;
		CHECK(cudaMemcpy(h_team, d_team, numBytes, cudaMemcpyHostToDevice));
	}

//	teams = new feedForward[numTrials][3];
	//run simulation
	while(generations < maxGens && catches < numTrials){//run contents of this loop in parallel
		catches = 0;

		for(int t = 0; t < numTrials;t++){

			//initialise teams
			for(int p = 0;p<numPreds;p++){
				feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
				Neuron** d_hidden;
				Create(*ff, predSubPops[p], hidden);
				CHECK(cudaMalloc(&d_hidden, hidden * sizeof(Neuron)));
				teams[t][p] = *ff;
				CHECK(cudaMemcpy(ff->HiddenUnits, d_hidden, hidden*sizeof(Neuron), cudaMemcpyHostToDevice));
			}
		}
		CHECK(cudaMemcpy(teams, d_teams, numBytes, cudaMemcpyHostToDevice));

//		PredatorPrey* pp;
		PredatorPrey* h_pp;
		State* d_state;
		Gridworld* d_world;
//		cudaMallocManaged(&pp, sizeof(PredatorPrey));
//		cudaMalloc((void**)&d_pp, sizeof(PredatorPrey));
		cudaMalloc(&d_state, sizeof(State));
		cudaMalloc(&d_world, sizeof(Gridworld));
		h_pp = newPredatorPrey(numPreds);
//		pp = newPredatorPrey(numPreds);
		reset(h_pp, numPreds);
//		reset(pp, numPreds);
		CHECK(cudaMemcpy(h_pp->state, d_state, sizeof(State), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(h_pp->world, d_world, sizeof(Gridworld), cudaMemcpyHostToDevice));
		//setup for kernel evaluation
		int inplen = getTotalInputs(teams[0][0]);
		int outlen = getTotalOutputs(teams[0][0]);
		double* input;
		CHECK(cudaMallocManaged(&input, inplen * sizeof(double)));
		double* output;
		CHECK(cudaMallocManaged(&output, outlen * sizeof(double)));

		//evaluate teams
		runEvaluationsParallel<<<blocks, threadsPerBlock>>>(d_state, d_world, d_teams, numPreds, input, output, inplen, outlen, trialsPerEval, sim, numTrials);
//		feedForward* t = evaluate(*pp, team, numPreds);

		//send memory back
		numBytes = numTrials * (numPreds * sizeof(feedForward));
		CHECK(cudaMemcpy(teams, d_teams, numBytes, cudaMemcpyDeviceToHost));
		for(int t = 0;t<numTrials;t++){
			numBytes = numPreds * sizeof(feedForward);
			CHECK(cudaMemcpy(h_team, d_teams[t], numBytes, cudaMemcpyDeviceToHost));
		}
		cudaDeviceSynchronize();
		//assign team scores
		//TODO: loop through all teams
		for(int n = 0; n < numTrials;n++){
			catches = catches + getCatches(teams[n][0]);
			if(bestFitness == 0 && !teamfound){
				bestFitness = getFitness(teams[n][0]);
			}

			printf("best fitness %d\n", bestFitness);
	//			printf("this team fitness: %d\n", getFitness(t[0]));

			//keep track of the best performing team
			if(getFitness(teams[n][0]) > bestFitness){
				bestFitness = getFitness(teams[n][0]);
				double* bestActivation = new double[teams[n]->numHidden];
				Neuron** bestNeurons = new Neuron*[teams[n]->numHidden];
				for(int i = 0;i<teams[n]->numHidden;i++){
					bestActivation[i] = teams[n]->Activation[i];
					bestNeurons[i] = teams[n]->HiddenUnits[i];
				}
				bestTeam = new feedForward;
				bestTeam[0].ID = teams[n][0].ID;
				bestTeam[0].Catches = teams[n][0].Catches;
				bestTeam[0].Fitness = teams[n][0].Fitness;
				bestTeam[0].GeneSize = teams[n][0].GeneSize;
				bestTeam[0].NumInputs = teams[n][0].NumInputs;
				bestTeam[0].NumOutputs = teams[n][0].NumOutputs;
				bestTeam[0].Parent1 = teams[n][0].Parent1;
				bestTeam[0].Parent2 = teams[n][0].Parent2;
				bestTeam[0].Trials = teams[n][0].Trials;
				bestTeam[0].bias = teams[n][0].bias;
				bestTeam[0].name = teams[n][0].name;
				bestTeam[0].numHidden = teams[n][0].numHidden;
				bestTeam[0].Activation=bestActivation;
				bestTeam[0].HiddenUnits = bestNeurons;
				//tag best team neurons
				for(int i = 0;i<numPreds;i++){
					Tag(bestTeam[0]);
				}
			}
			//if this is the first run, take the team as the baseline best team
			if(!teamfound){
				teamfound = true;
				double* bestActivation = new double[teams[n]->numHidden];
				Neuron** bestNeurons = new Neuron*[teams[n]->numHidden];
				for(int i = 0;i<teams[n]->numHidden;i++){
					bestActivation[i] = teams[n]->Activation[i];
					bestNeurons[i] = teams[n]->HiddenUnits[i];
				}
				bestTeam = new feedForward;
				bestTeam[0].ID = teams[n][0].ID;
				bestTeam[0].Catches = teams[n][0].Catches;
				bestTeam[0].Fitness = teams[n][0].Fitness;
				bestTeam[0].GeneSize = teams[n][0].GeneSize;
				bestTeam[0].NumInputs = teams[n][0].NumInputs;
				bestTeam[0].NumOutputs = teams[n][0].NumOutputs;
				bestTeam[0].Parent1 = teams[n][0].Parent1;
				bestTeam[0].Parent2 = teams[n][0].Parent2;
				bestTeam[0].Trials = teams[n][0].Trials;
				bestTeam[0].bias = teams[n][0].bias;
				bestTeam[0].name = teams[n][0].name;
				bestTeam[0].numHidden = teams[n][0].numHidden;
				bestTeam[0].Activation=bestActivation;
				bestTeam[0].HiddenUnits = bestNeurons;
			}
		}

		printf("Generation %d, best fitness is %d, catches is %d\n", generations+1, bestFitness, catches);

		//check for stagnation and burst mutate if stagnated
		if(generations%burstGens == 0 && generations != 0){
			//burst mutate
			stagnated = true;

			for(int pred = 0; pred < numPreds; pred++){
				Population* predPop = predSubPops[pred];
				for(int i = 0; i< hidden;i++){
					Population subpop = predPop[i];
					for(int n = 0; n< numIndivs;n++){
						Neuron* indiv = subpop.Individuals[n];
						Neuron** hid = getHiddenUnits(bestTeam[0]);
//						if(n ==19){
//							n = 19;
//						}
						subpop.Individuals[n] = perturb(indiv, *hid[i], bestTeam->GeneSize);
					}
				}
			}
		}
		//sort by fitness, mate upper part and mutate populations if not stagnated
		if(!stagnated){
			for(int i = 0 ;i<numPreds;i++){
				for(int j = 0;j<hidden;j++){
					predSubPops[i][j] = sortNeurons(predSubPops[i][j]);
					predSubPops[i][j] = mate(predSubPops[i][j]);
					predSubPops[i][j] = mutate(predSubPops[i][j], mutationRate);
				}
			}
		}
		stagnated = false;
		generations++;
		CHECK(cudaFree(d_teams));
		CHECK(cudaFree(d_state));
		CHECK(cudaFree(d_world));

	}
}


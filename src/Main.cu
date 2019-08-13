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
Neuron* bestTeam;
int bestGene;
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

struct team{
	int numOutputs;
	int numInputs;
	double act1[15];
	Neuron t1[15];
	double act2[15];
	Neuron t2[15];
	double act3[15];
	Neuron t3[15];
	int fitness;
	int numHidden;
	int catches;
};

struct teamArr{
	team team;
};


Population* init(int hid, int num, int genes){
	Population* pops = new Population[hid];
	for(int i = 0; i < hid; i++){
		Population* p = newPopulation(num, genes);
		createIndividuals(p);
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
/*
__global__ void runEvaluationsParallel(State* statepntr, Gridworld* worldpntr, teamArr* teams, int numPreds, double* input, double* output, int inplen, int outlen, int trialsPerEval, bool sim, int numTrials){
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
					double* out = Activate(teams[i].teams[pred], input, inplen, output);
					PerformPredatorAction(statepntr, worldpntr, pred, out, teams[i].teams[pred].NumOutputs);//change to use state?
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
			teams[i].teams[pred].Fitness = (total_fitness); // /trialsPerEval
			teams[i].teams[pred].Catches = catches;
			Neuron** d_neurons;
			// <<<blocks, threadsPerBlock>>>
			int numBytes = teams[i].teams[pred].numHidden * sizeof(teams[i].teams[pred].HiddenUnits[0]);
			//case 1
	//		cudaMalloc(&d_neurons, numBytes);//optimise to only take neuron fitness and trials not whole struct
	//		cudaMemcpy(team[pred].HiddenUnits, d_neurons, numBytes, cudaMemcpyHostToDevice);
	//		kernelAssignFitness<<<1, team[pred].numHidden>>>(total_fitness, d_neurons);
	//		cudaDeviceSynchronize();
	//		cudaMemcpy(team[pred].HiddenUnits, d_neurons, numBytes, cudaMemcpyDeviceToHost);
			//case 2
	//		kernelAssignFitness<<<1, team[pred].numHidden>>>(total_fitness, team[pred].HiddenUnits);
	//		cudaDeviceSynchronize();
			for(int i = 0; i<teams[i].teams[pred].numHidden;i++){
				Neuron n = teams[i].teams[pred].HiddenUnits[i];
				n.Fitness = teams[i].teams[pred].Fitness;
				n.Trials++;
				teams[i].teams[pred].HiddenUnits[i] = n;
			}
		}
//		printf("Team %d's fitness: %d\n",i , teams[i][0].Fitness);
	}
//	return team;
}*/

void CHECK(cudaError_t err){
	if(err){
		printf("Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
}


__global__ void testKernel(teamArr* teams){
	int index = threadIdx.x;
	double test = teams[index].team.t1[0].Weight[0];
	teams[index].team.fitness = index;
//	printf("Index %d Weight %d\n", index, test);
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

//	feedForward teams[numTrials][numPreds];
	teamArr* teams;
//	feedForward** d_teams;
	teamArr* d_teams;
	team h_team;
	team d_team;
	int numBytes = numTrials * sizeof(team);
//	cudaMallocManaged(&teams, numBytes);
	CHECK(cudaMalloc((void **)&d_teams, numBytes));
	teams = (teamArr*)malloc(numBytes);
//	teams = new teamArr();
	for (int t =0;t < numTrials;t++){
		numBytes = sizeof(team);
		CHECK(cudaMalloc((void**)&teams[t].team, numBytes));

//		h_team = new team();
//		teams.teams = h_team;
//		d_teams.teams = d_team;
//		cudaMallocManaged(&team, numBytes);
//		teams[t] = h_team;
//		d_teams[t] = d_team;
//		CHECK(cudaMemcpy(h_team, d_team, numBytes, cudaMemcpyHostToDevice));
	}

//	teams = new feedForward[numTrials][3];
	//run simulation
	while(generations < maxGens && catches < numTrials){//run contents of this loop in parallel
		catches = 0;
		feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
		for(int t = 0; t < numTrials;t++){
			for(int p = 0;p<numPreds;p++){
				ff[p] = Create(ff[p], predSubPops[p], hidden);
			}
			team tm;
			for(int i = 0;i<hidden;i++){
				tm.act1[i] = ff[0].Activation[i];
				tm.act2[i] = ff[1].Activation[i];
				tm.act3[i] = ff[2].Activation[i];
				tm.t1[i] = ff[0].HiddenUnits[i];
				tm.t2[i] = ff[1].HiddenUnits[i];
				tm.t3[i] = ff[2].HiddenUnits[i];
			}
			tm.catches = ff->Catches;
			tm.fitness = ff->Fitness;
			tm.numHidden = ff->numHidden;
			tm.numInputs = ff->NumInputs;
			tm.numOutputs = ff->NumOutputs;
			teams[t].team = tm;
			//copy everything to GPU struct
			CHECK(cudaMemcpy(&d_teams[t].team, &teams[t].team, sizeof(team), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_teams[t].team.t1, &teams[t].team.t1, sizeof(teams[t].team.t1), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_teams[t].team.t2, &teams[t].team.t2, sizeof(teams[t].team.t2), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_teams[t].team.t3, &teams[t].team.t3, sizeof(teams[t].team.t3), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_teams[t].team.act1, &teams[t].team.act1, sizeof(teams[t].team.act1), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_teams[t].team.act2, &teams[t].team.act2, sizeof(teams[t].team.act2), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_teams[t].team.act3, &teams[t].team.act3, sizeof(teams[t].team.act3), cudaMemcpyHostToDevice));
//			CHECK(cudaMemcpy(&(d_teams->team), &tm, sizeof(team), cudaMemcpyHostToDevice));
//			free(ff);
			//initialise teams
//			for(int p = 0;p<numPreds;p++){
//				feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
//				Neuron** d_hidden;
//				Create(*ff, predSubPops[p], hidden);
//				CHECK(cudaMalloc(&d_hidden, hidden * sizeof(Neuron)));
//				teams[t].teams[p] = *ff;
//				CHECK(cudaMemcpy(ff->HiddenUnits, d_hidden, hidden*sizeof(Neuron), cudaMemcpyHostToDevice));
//			}
		}
		int numBytes = numTrials * sizeof(team);
		CHECK(cudaMemcpy(d_teams, teams, numBytes, cudaMemcpyHostToDevice));

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
		CHECK(cudaMemcpy(d_state, h_pp->state, sizeof(State), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_world, h_pp->world,  sizeof(Gridworld), cudaMemcpyHostToDevice));
		//setup for kernel evaluation
//		int inplen = getTotalInputs(teams[0][0]);
		int inplen = (teams[0].team.numInputs);
//		int outlen = getTotalOutputs(teams[0][0]);
		int outlen = (teams[0].team.numOutputs);
		double* input;
		CHECK(cudaMallocManaged(&input, inplen * sizeof(double)));
		double* output;
		CHECK(cudaMallocManaged(&output, outlen * sizeof(double)));

		//evaluate teams
		testKernel<<<1, 100>>>(d_teams);
//		runEvaluationsParallel<<<blocks, threadsPerBlock>>>(d_state, d_world, d_teams, numPreds, input, output, inplen, outlen, trialsPerEval, sim, numTrials);
//		feedForward* t = evaluate(*pp, team, numPreds);

		cudaDeviceSynchronize();
		//send memory back
		numBytes = numTrials * sizeof(team);
		CHECK(cudaMemcpy(teams, d_teams, numBytes, cudaMemcpyDeviceToHost));
		for(int t = 0;t<numTrials;t++){
			CHECK(cudaMemcpy(&teams[t].team, &d_teams[t].team, sizeof(team), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&teams[t].team.t1, &d_teams[t].team.t1, sizeof(teams[t].team.t1), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&teams[t].team.t2, &d_teams[t].team.t2, sizeof(teams[t].team.t2), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&teams[t].team.t3, &d_teams[t].team.t3, sizeof(teams[t].team.t3), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&teams[t].team.act1, &d_teams[t].team.act1, sizeof(teams[t].team.act1), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&teams[t].team.act2, &d_teams[t].team.act2, sizeof(teams[t].team.act2), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(&teams[t].team.act3, &d_teams[t].team.act3, sizeof(teams[t].team.act3), cudaMemcpyDeviceToHost));

		}
//		for(int t = 0;t<numTrials;t++){
//			numBytes = sizeof(team);
//			CHECK(cudaMemcpy(h_team, d_teams[t].team, numBytes, cudaMemcpyDeviceToHost));
//		}
		//assign team scores
		//TODO: loop through all teams
		for(int n = 0; n < numTrials;n++){
			catches = catches + (teams[n].team.catches);
			if(bestFitness == 0 && !teamfound){
				bestFitness = (teams[n].team.fitness);
			}

//			printf("best fitness %d\n", bestFitness);
	//			printf("this team fitness: %d\n", getFitness(t[0]));

			//keep track of the best performing team
			if((teams[n].team.fitness) > bestFitness){
				bestFitness = (teams[n].team.fitness);
				bestGene = teams[n].team.numInputs + teams[n].team.numOutputs;
//				double* bestActivation = new double[teams[n].teams[0].numHidden];
//				Neuron* bestNeurons = new Neuron[teams[n].teams[0].numHidden];
//				for(int i = 0;i<teams[n].teams[0].numHidden;i++){
//					bestTeam[0].Activation[i] = teams[n].teams[0].act1[i];
//					bestTeam[0].HiddenUnits[i] = teams[n].teams[0].t1[i];
//				}
//				bestTeam = new feedForward;
//				bestTeam[0].ID = teams[n].teams[0].ID;
//				bestTeam[0].Catches = teams[n].teams[0].Catches;
//				bestTeam[0].Fitness = teams[n].teams[0].Fitness;
//				bestTeam[0].GeneSize = teams[n].teams[0].GeneSize;
//				bestTeam[0].NumInputs = teams[n].teams[0].NumInputs;
//				bestTeam[0].NumOutputs = teams[n].teams[0].NumOutputs;
//				bestTeam[0].Parent1 = teams[n].teams[0].Parent1;
//				bestTeam[0].Parent2 = teams[n].teams[0].Parent2;
//				bestTeam[0].Trials = teams[n].teams[0].Trials;
//				bestTeam[0].bias = teams[n].teams[0].bias;
//				bestTeam[0].name = teams[n].teams[0].name;
				bestTeam = teams[n].team.t1;
//				bestTeam[0].Activation=bestActivation;
//				bestTeam[0].HiddenUnits = bestNeurons;
				//tag best team neurons
//				for(int i = 0;i<numPreds;i++){
//					Tag(bestTeam[0]);
//				}
			}
			//if this is the first run, take the team as the baseline best team
			if(!teamfound){
				teamfound = true;
				bestFitness = (teams[n].team.fitness);
				bestGene = teams[n].team.numInputs + teams[n].team.numOutputs;
//				double* bestActivation = new double[teams[n].teams[0].numHidden];
//				Neuron* bestNeurons = new Neuron[teams[n].teams[0].numHidden];
//				for(int i = 0;i<teams[n].teams[0].numHidden;i++){
//					bestTeam[0].Activation[i] = teams[n].teams[0].Activation[i];
//					bestTeam[0].HiddenUnits[i] = teams[n].teams[0].HiddenUnits[i];
//				}
//				bestTeam = new feedForward;
//				bestTeam[0].ID = teams[n].teams[0].ID;
//				bestTeam[0].Catches = teams[n].teams[0].Catches;
//				bestTeam[0].Fitness = teams[n].teams[0].Fitness;
//				bestTeam[0].GeneSize = teams[n].teams[0].GeneSize;
//				bestTeam[0].NumInputs = teams[n].teams[0].NumInputs;
//				bestTeam[0].NumOutputs = teams[n].teams[0].NumOutputs;
//				bestTeam[0].Parent1 = teams[n].teams[0].Parent1;
//				bestTeam[0].Parent2 = teams[n].teams[0].Parent2;
//				bestTeam[0].Trials = teams[n].teams[0].Trials;
//				bestTeam[0].bias = teams[n].teams[0].bias;
//				bestTeam[0].name = teams[n].teams[0].name;
				bestTeam = teams[n].team.t1;
//				bestTeam[0].Activation = teams[n].teams[0].Activation;
//				bestTeam[0].HiddenUnits = *bestNeurons;
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
						Neuron indiv = subpop.Individuals[n];
						Neuron* hid = bestTeam;
//						if(n ==19){
//							n = 19;
//						}
						subpop.Individuals[n] = perturb(indiv, hid[i], bestGene);
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


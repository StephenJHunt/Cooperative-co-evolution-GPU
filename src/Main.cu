
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

#ifndef nPreds
#define nPreds = 3
#endif
#ifndef nHidden
#define nHidden = 15
#endif
//globals
Neuron* bestTeam;
int bestGene;
Population* subPops;
Population** predSubPops;
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

//struct aTeam{//moved to feedForward.h
//	int numOutputs;
//	int numInputs;
//	double act1[15];
//	Neuron t1[15];
//	double act2[15];
//	Neuron t2[15];
//	double act3[15];
//	Neuron t3[15];
//	int fitness;
//	int numHidden;
//	int catches;
//};

struct teamArr{
	aTeam team;
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

int h_calculateDistance(Gridworld* h_world, int h_predX, int h_predY, int h_preyX, int h_preyY){
	double h_xDist = 0;
	double h_yDist = 0;

	h_xDist = abs((double)(h_predX-h_preyX));
	if(h_xDist > double(h_world->length/2)){
		h_xDist = double(h_world->length) - h_xDist;
	}

	h_yDist = abs((double)(h_predY-h_preyY));
		if(h_yDist > double(h_world->height/2)){
			h_yDist = double(h_world->height) - h_yDist;
		}
	return int(h_xDist + h_yDist);
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

teamArr* h_eval(Gridworld* h_worldpntr, teamArr* h_teams, int h_numPreds, double* h_input, double* h_output, int h_inplen, int h_outlen, int h_trialsPerEval, bool h_sim, int h_numTrials){
	State* h_statepntr = new State();
	h_reset(h_statepntr, h_numPreds);

	int h_catches = 0;
	int h_totalfitness = 0;

	int h_PreyPositions[2][9] = {{16, 50, 82, 82, 82, 16, 50, 50, 82},{50, 50, 50, 82, 16, 50, 16, 82, 50}};

	for(int i=0;i<h_trialsPerEval;i++){
		int h_fitness = 0;
		int h_steps = 0;
		int h_maxSteps = 150;
		int h_avg_init_dist = 0;
		int h_avg_final_dist = 0;

		State h_state;
		Gridworld h_world;

		h_setPreyPosition(h_statepntr, h_PreyPositions[0][i], h_PreyPositions[1][i]);
		h_state = *h_statepntr;
		h_world = *h_worldpntr;

		int h_nearestDist = 100;
		int h_nearestPred = 0;
		int h_currentDist = 0;


	}
	return h_teams;
}

__global__ void runEvaluationsParallel(Gridworld* worldpntr, teamArr* d_teams, int numPreds, double* input, double* output, int inplen, int outlen, int trialsPerEval, bool sim, int numTrials){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	State* statepntr = new State();
	kernelReset(statepntr, numPreds);
//
	for(int i = index;i < numTrials;i+= stride){
		int catches = 0;
		int total_fitness = 0;
//
		int PreyPositions[2][9] = {{16, 50, 82, 82, 82, 16, 50, 50, 82},{50, 50, 50, 82, 16, 50, 16, 82, 50}};
//
		for(int l = 0;l < trialsPerEval;l++){//parallel?
			int fitness =0;
			int steps = 0;
			int maxSteps = 150;
			int avg_init_dist = 0;
			int avg_final_dist = 0;

			State state;
			Gridworld world;

			setPreyPosition(statepntr, PreyPositions[0][l], PreyPositions[1][l]);//use state instead of PredatorPrey?
			state = *statepntr;
			world = *worldpntr;

			int nearestDist = 100;//so that closest pred changes
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
					double* out = Activate(&d_teams[i].team, input, inplen, output);
					PerformPredatorAction(statepntr, worldpntr, pred, out, d_teams[i].team.numOutputs);//change to use state?
				}
				steps++;
			}
			if(Caught(statepntr)){
				if(sim == true){
					printf("Simulation Complete\n");
					printf("Predator at position %d, %d caught the prey at position %d, %d after %d steps", statepntr->PredatorX[nearestPred], statepntr->PredatorY[nearestPred], statepntr->PreyX, statepntr->PreyY, steps);
				}
			}

			for(int p = 0; p < numPreds;p++){
				avg_final_dist = avg_final_dist + calculateDistance(worldpntr, statepntr->PredatorX[p], statepntr->PredatorY[p], statepntr->PreyX, statepntr->PreyY);
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

		d_teams[i].team.fitness = total_fitness; // /trialsPerEval
		d_teams[i].team.catches = catches;

		for(int i2 = 0; i2<d_teams[i].team.numHidden;i2++){
			Neuron n1 = d_teams[i].team.t1[i2];
			Neuron n2 = d_teams[i].team.t2[i2];
			Neuron n3 = d_teams[i].team.t3[i2];
			n1.Fitness = d_teams[i].team.fitness;
			n2.Fitness = d_teams[i].team.fitness;
			n3.Fitness = d_teams[i].team.fitness;
			n1.Trials++;
			n2.Trials++;
			n3.Trials++;
			d_teams[i].team.t1[i2] = n1;
			d_teams[i].team.t2[i2] = n2;
			d_teams[i].team.t3[i2] = n3;
		}
	}
}

void CHECK(cudaError_t err){
	if(err){
		printf("Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
}


__global__ void testKernel(teamArr* teams, double* input, int inplen){
	int index = threadIdx.x;
//	input = new double[inplen];
	input[0] = 1.0;
//	double test = teams[index].team.t1[0].Weight[0];
//	teams[index].team.fitness = index+1;
	for(int i2 = 0; i2<teams[index].team.numHidden;i2++){
		Neuron n1 = teams[index].team.t1[i2];
		Neuron n2 = teams[index].team.t2[i2];
		Neuron n3 = teams[index].team.t3[i2];
		n1.Fitness = teams[index].team.fitness;
		n2.Fitness = teams[index].team.fitness;
		n3.Fitness = teams[index].team.fitness;
		n1.Trials++;
		n2.Trials++;
		n3.Trials++;
		teams[index].team.t1[i2] = n1;
		teams[index].team.t2[i2] = n2;
		teams[index].team.t3[i2] = n3;
		}
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

	teamArr* teams;
	teamArr* d_teams;

	//run simulation
	while(generations < maxGens && catches < numTrials){//run contents of this loop in parallel
		int numTeamBytes = numTrials * sizeof(aTeam);
		CHECK(cudaMalloc(&d_teams, numTeamBytes));
		teams = (teamArr*)malloc(numTeamBytes);
		catches = 0;
		feedForward* ff = newFeedForward(numInputs, hidden, numOutputs, false);
		for(int t = 0; t < numTrials;t++){
			for(int p = 0;p<numPreds;p++){
				ff[p] = Create(ff[p], predSubPops[p], hidden);
			}
			aTeam tm;
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
		}
		CHECK(cudaMemcpy(d_teams, teams, numTeamBytes, cudaMemcpyHostToDevice));//State is causing the issue

		PredatorPrey* h_pp;
//		State* d_state;
		Gridworld* d_world;
//		CHECK(cudaMalloc(&d_state, sizeof(State)));
		CHECK(cudaMalloc(&d_world, sizeof(Gridworld)));
		h_pp = newPredatorPrey(numPreds);
		reset(h_pp, numPreds);
//		CHECK(cudaMemcpy(d_state, h_pp->state, sizeof(State), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_world, h_pp->world,  sizeof(Gridworld), cudaMemcpyHostToDevice));
		//setup for kernel evaluation
		int inplen = (teams[0].team.numInputs);
		int outlen = (teams[0].team.numOutputs);
		double* d_input;
		double* h_input;
		CHECK(cudaMalloc(&d_input, inplen * sizeof(double)));
		h_input = (double*)malloc(inplen * sizeof(double));
		double* d_output;
		double* h_output;
		CHECK(cudaMalloc(&d_output, outlen * sizeof(double)));
		h_output = (double*)malloc(outlen * sizeof(double));

		//evaluate teams
//		testKernel<<<1, 100>>>(d_teams, d_input, inplen);
		// blocks, threadsPerBlock
//		runEvaluationsParallel<<<blocks, threadsPerBlock>>>(d_world, d_teams, numPreds, d_input, d_output, inplen, outlen, trialsPerEval, sim, numTrials);
//		feedForward* t = evaluate(*pp, team, numPreds);
		CHECK(cudaPeekAtLastError());
//		cudaDeviceSynchronize();
		//send memory back
		CHECK(cudaMemcpy(teams, d_teams, numTeamBytes, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_output, d_output, outlen * sizeof(double), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_input, d_input, inplen * sizeof(double), cudaMemcpyDeviceToHost));

		//assign team scores
		//TODO: loop through all teams
		for(int n = 0; n < numTrials;n++){
			catches = catches + (teams[n].team.catches);
			if(bestFitness == 0 && !teamfound){
				bestFitness = (teams[n].team.fitness);
			}

			//keep track of the best performing team
			if((teams[n].team.fitness) > bestFitness){
				bestFitness = (teams[n].team.fitness);
				bestGene = teams[n].team.numInputs + teams[n].team.numOutputs;
				bestTeam = teams[n].team.t1;
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
				bestTeam = teams[n].team.t1;
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
		free(teams);
//		CHECK(cudaFree(d_state));
		CHECK(cudaFree(d_world));
		CHECK(cudaFree(d_input));
		CHECK(cudaFree(d_output));
		free(h_input);
		free(h_output);

	}
}


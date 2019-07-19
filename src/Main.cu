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
#include "environment.h"
#include "network.h"
#include "population.h"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

//globals
Network* bestTeam;
Population* subPops;
Population** predSubPops;
Gridworld world;

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

Network* evaluate(PredatorPrey e, feedForward* team){
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
	tempState tmpstate;

	tempState* states;

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
		//stuff
	}
}

int main(int argc, char **argv)
{


//	char* test = "hello";//this is a string now
	runTests();
	//printf(test);
	//printf("\n");
    //printf("Hello World!\n");
    //CUDAHello<<<1,10>>>();
//    cudaDeviceReset();
    //
}


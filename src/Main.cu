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


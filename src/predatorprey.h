#ifndef PREDATORPREY_H_
#define PREDATORPREY_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//include other files
#include "environment.h"

bool caught = false;

struct PredatorPrey{
	char* name;
	State* state;
	Gridworld* world;
};

PredatorPrey* newPredatorPrey(){
	return new PredatorPrey{"Predator Prey Task", new State, new Gridworld};
}

void reset(PredatorPrey pp, int n){
	pp.world->length = 100;
	pp.world->height = 100;

	pp.state->PreyX = 50;
	pp.state->PreyY = 50;

	for(int i=0;i<n;i++){
		pp.state->PredatorX[i] = i*2;
		pp.state->PredatorY[i] = 0;
	}
	caught = false;
}

int getMaxPosition(double* action, int actionlen){
 double max = action[0];
 int result = 0;
 for(int i = 0;i < actionlen;i++){
	 if(action[i] > max){
		 max = action[i];
		 result = i;
	 }
 }
 return result;
}

void PerformPredatorAction(PredatorPrey pp, int pos, double* action, int actionlen){
	int predAction = getMaxPosition(action, actionlen);

	//possible movements. NESW in order
	if(predAction == 0){
		pp.state->PredatorY[pos]++;
	}
	else if(predAction == 1){
		pp.state->PredatorX[pos]++;
	}
	else if(predAction == 2){
		pp.state->PredatorY[pos]--;
	}
	else if(predAction == 3){
		pp.state->PredatorX[pos]--;
	}

	//wrap around world
	if(pp.state->PredatorX[pos] > pp.world->length){
		pp.state->PredatorX[pos] = pp.state->PredatorX[pos] - pp.world->length;
	}
	if(pp.state->PredatorY[pos] > pp.world->height){
		pp.state->PredatorY[pos] = pp.state->PredatorY[pos] - pp.world->height;
	}
	if(pp.state->PredatorX[pos] < 0){
		pp.state->PredatorX[pos] = pp.state->PredatorX[pos] + pp.world->length;
	}
	if(pp.state->PredatorY[pos] < 0){
		pp.state->PredatorY[pos] = pp.state->PredatorY[pos] + pp.world->height;
	}

	//is the predator at the same pos as the prey
	if((pp.state->PredatorX[pos] == pp.state->PreyX) && (pp.state->PredatorY[pos] == pp.state->PreyY)){
		caught = true;
	}
}

void PerformPreyAction(PredatorPrey pp, int nearest){
	double xDistance = (double)pp.state->PredatorX[nearest] - pp.state->PreyX;
	if(abs(xDistance) > (double)pp.world->length/2){

	}
}











#endif

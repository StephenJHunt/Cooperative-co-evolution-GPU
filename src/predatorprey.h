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

PredatorPrey* newPredatorPrey(int numPreds){
	State* st = new State{{0,0,0}, {0,0,0}, 0, 0};
	return new PredatorPrey{"Predator Prey Task", st, new Gridworld};
}

void reset(PredatorPrey* pp, int n){
	pp->world->length = 100;
	pp->world->height = 100;

	pp->state->PreyX = 50;
	pp->state->PreyY = 50;

	for(int i=0;i<n;i++){
		if(i>0) pp->state->PredatorX[i] = (i*2)-1 ;//-1 // (99/n)
		else pp->state->PredatorX[i] = 0;
		pp->state->PredatorY[i] = 0;//99
	}
	pp->state->Caught = false;
}

__device__ void kernelReset(State* st, int n){
	for(int i = 0;i<n;i++){
		if(i>0)st->PredatorX[i] = (i*2)-1;
		else st->PredatorX[i] = 0;
		st->PredatorY[i]=0;
	}
	st->Caught = false;
}

__device__ void setPreyPosition(State* state,int x, int y){
	state->PreyX = x;
	state->PreyY = y;
}

__device__ int getMaxPosition(double* action, int actionlen){
 double max = 0;
 int result = 0;
// int n = 0;
// int e = 0;
// int s = 0;
// int w = 0;
// for(int i = 0;i<actionlen;i++){
//	 if(action[i] == 0)n++;
//	 if(action[i] == 1)e++;
//	 if(action[i] == 2)s++;
//	 if(action[i] == 3)w++;
// }
// if(n > e && n > s && n > w)return 0;
// if(e > n && e > s && e > w)return 1;
// if(s > n && s > e && s > w)return 2;
// if(w > n && w > e && w > s)return 3;
 for(int i = 0;i < actionlen-1;i++){
	 if(action[i] > max){
		 max = action[i];
		 result = i;
	 }
 }
 return result;
}

__device__ void PerformPredatorAction(State* state, Gridworld* world, int pos, double* action, int actionlen){
	int predAction = getMaxPosition(action, actionlen);
//	printf("predaction:%d\n", predAction);
	//possible movements. NESW in order
	if(predAction == 0){
		state->PredatorY[pos]++;
	}
	else if(predAction == 1){
		state->PredatorX[pos]++;
	}
	else if(predAction == 2){
		state->PredatorY[pos]--;
	}
	else if(predAction == 3){
		state->PredatorX[pos]--;
	}

	//wrap around world
	if(state->PredatorX[pos] >= world->length){
		state->PredatorX[pos] = state->PredatorX[pos] - world->length;

//		pp.state->PredatorX[pos] = pp.state->PredatorX[pos]-1;
//
//		if(pp.state->PreyY < pp.state->PredatorY[pos]){//prey below
//			pp.state->PredatorY[pos]--;
//		}
//		if(pp.state->PreyY > pp.state->PredatorY[pos]){//prey above
//			pp.state->PredatorY[pos]++;
//		}

	}
	if(state->PredatorY[pos] >= world->height){
		state->PredatorY[pos] = state->PredatorY[pos] - world->height;

//		pp.state->PredatorY[pos] = pp.state->PredatorY[pos]-1;
//
//		if(pp.state->PreyX < pp.state->PredatorX[pos]){//prey left
//			pp.state->PredatorX[pos]--;
//		}
//		if(pp.state->PreyX > pp.state->PredatorX[pos]){//prey right
//			pp.state->PredatorX[pos]++;
//		}
	}
	if(state->PredatorX[pos] < 0){
		state->PredatorX[pos] = state->PredatorX[pos] + world->length;

//		pp.state->PredatorX[pos] = pp.state->PredatorX[pos]+1;
//
//		if(pp.state->PreyY < pp.state->PredatorY[pos]){//prey below
//			pp.state->PredatorY[pos]--;
//		}
//		if(pp.state->PreyY > pp.state->PredatorY[pos]){//prey above
//			pp.state->PredatorY[pos]++;
//		}
	}
	if(state->PredatorY[pos] < 0){
		state->PredatorY[pos] = state->PredatorY[pos] + world->height;
//
//		pp.state->PredatorY[pos] = pp.state->PredatorY[pos]+1;
//
//		if(pp.state->PreyX < pp.state->PredatorX[pos]){//prey left
//			pp.state->PredatorX[pos]--;
//		}
//		if(pp.state->PreyX > pp.state->PredatorX[pos]){//prey right
//			pp.state->PredatorX[pos]++;
//		}
	}

	//is the predator at the same pos as the prey
	if((state->PredatorX[pos] == state->PreyX) && (state->PredatorY[pos] == state->PreyY)){
		state->Caught = true;
	}
}

__device__ void PerformPreyAction(State* state, Gridworld* world, int nearest){
	double xDistance = (double)(state->PredatorX[nearest] - state->PreyX);
	if(abs(xDistance) > (double)(world->length/2)){
		double temp = xDistance;
		xDistance = (double)(world->length - abs(xDistance));
		if(temp > 0){
			xDistance = 0- xDistance;
		}
	}

	double yDistance = (double)(state->PredatorY[nearest] - state->PreyY);
	if(abs(yDistance) > (double)(world->height/2)){
		double temp = yDistance;
		yDistance = (double)(world->height - abs(yDistance));
		if(temp > 0){
			yDistance = 0- yDistance;
		}
	}

	//NESW movement
	if(yDistance < 0 && (abs((double)(yDistance)) >= abs((double)xDistance))){
		state->PreyY++;
	}
	else if(xDistance < 0 && (abs((double)xDistance) >= abs((double)yDistance))){
		state->PreyX++;
	}
	else if(yDistance > 0 && (abs((double)(yDistance)) >= abs((double)xDistance))){
		state->PreyY--;
	}
	else if(xDistance > 0 && (abs((double)(xDistance)) >= abs((double)yDistance))){
		state->PreyX--;
	}

	if(state->PreyX >= world->length){
		state->PreyX = state->PreyX - world->length;
//		pp.state->PreyX = pp.state->PreyX -1;
	}
	if(state->PreyY >= world->height){
		state->PreyY = state->PreyY - world->height;
//		pp.state->PreyY = pp.state->PreyY-1;
	}
	if(state->PreyX < 0){
		state->PreyX = state->PreyX + world->length;
//		pp.state->PreyX = pp.state->PreyX+1;
	}
	if(state->PreyY < 0){
		state->PreyY = state->PreyY + world->height;
//		pp.state->PreyY = pp.state->PreyY+1;
	}

}

State* getState(PredatorPrey pp){
	return pp.state;
}

Gridworld* getWorld(PredatorPrey pp){
	return pp.world;
}

__device__ bool Caught(State* state){
	return state->Caught;
}



#endif

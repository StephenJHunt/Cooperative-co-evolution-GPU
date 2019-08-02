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
	State* st = new State{new int[numPreds], new int[numPreds], 0, 0};
	return new PredatorPrey{"Predator Prey Task", st, new Gridworld};
}

void reset(PredatorPrey pp, int n){
	pp.world->length = 100;
	pp.world->height = 100;

	pp.state->PreyX = 50;
	pp.state->PreyY = 50;

	for(int i=0;i<n;i++){
		if(i>0) pp.state->PredatorX[i] = (i*2)-1 ;//-1 // (99/n)
		else pp.state->PredatorX[i] = 0;
		pp.state->PredatorY[i] = 0;//99
	}
	pp.state->Caught = false;
}

__device__ void kernelReset(PredatorPrey pp, int n){

}

void setPreyPosition(PredatorPrey pp,int x, int y){
	pp.state->PreyX = x;
	pp.state->PreyY = y;
}

int getMaxPosition(double* action, int actionlen){
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

void PerformPredatorAction(PredatorPrey pp, int pos, double* action, int actionlen){
	int predAction = getMaxPosition(action, actionlen);
//	printf("predaction:%d\n", predAction);
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
	if(pp.state->PredatorX[pos] >= pp.world->length){
		pp.state->PredatorX[pos] = pp.state->PredatorX[pos] - pp.world->length;

//		pp.state->PredatorX[pos] = pp.state->PredatorX[pos]-1;
//
//		if(pp.state->PreyY < pp.state->PredatorY[pos]){//prey below
//			pp.state->PredatorY[pos]--;
//		}
//		if(pp.state->PreyY > pp.state->PredatorY[pos]){//prey above
//			pp.state->PredatorY[pos]++;
//		}

	}
	if(pp.state->PredatorY[pos] >= pp.world->height){
		pp.state->PredatorY[pos] = pp.state->PredatorY[pos] - pp.world->height;

//		pp.state->PredatorY[pos] = pp.state->PredatorY[pos]-1;
//
//		if(pp.state->PreyX < pp.state->PredatorX[pos]){//prey left
//			pp.state->PredatorX[pos]--;
//		}
//		if(pp.state->PreyX > pp.state->PredatorX[pos]){//prey right
//			pp.state->PredatorX[pos]++;
//		}
	}
	if(pp.state->PredatorX[pos] < 0){
		pp.state->PredatorX[pos] = pp.state->PredatorX[pos] + pp.world->length;

//		pp.state->PredatorX[pos] = pp.state->PredatorX[pos]+1;
//
//		if(pp.state->PreyY < pp.state->PredatorY[pos]){//prey below
//			pp.state->PredatorY[pos]--;
//		}
//		if(pp.state->PreyY > pp.state->PredatorY[pos]){//prey above
//			pp.state->PredatorY[pos]++;
//		}
	}
	if(pp.state->PredatorY[pos] < 0){
		pp.state->PredatorY[pos] = pp.state->PredatorY[pos] + pp.world->height;
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
	if((pp.state->PredatorX[pos] == pp.state->PreyX) && (pp.state->PredatorY[pos] == pp.state->PreyY)){
		caught = true;
	}
}

void PerformPreyAction(PredatorPrey pp, int nearest){
	double xDistance = (double)(pp.state->PredatorX[nearest] - pp.state->PreyX);
	if(abs(xDistance) > (double)(pp.world->length/2)){
		double temp = xDistance;
		xDistance = (double)(pp.world->length - abs(xDistance));
		if(temp > 0){
			xDistance = 0- xDistance;
		}
	}

	double yDistance = (double)(pp.state->PredatorY[nearest] - pp.state->PreyY);
	if(abs(yDistance) > (double)(pp.world->height/2)){
		double temp = yDistance;
		yDistance = (double)(pp.world->height - abs(yDistance));
		if(temp > 0){
			yDistance = 0- yDistance;
		}
	}

	//NESW movement
	if(yDistance < 0 && (abs((double)(yDistance)) >= abs((double)xDistance))){
		pp.state->PreyY++;
	}
	else if(xDistance < 0 && (abs((double)xDistance) >= abs((double)yDistance))){
		pp.state->PreyX++;
	}
	else if(yDistance > 0 && (abs((double)(yDistance)) >= abs((double)xDistance))){
		pp.state->PreyY--;
	}
	else if(xDistance > 0 && (abs((double)(xDistance)) >= abs((double)yDistance))){
		pp.state->PreyX--;
	}

	if(pp.state->PreyX >= pp.world->length){
		pp.state->PreyX = pp.state->PreyX - pp.world->length;
//		pp.state->PreyX = pp.state->PreyX -1;
	}
	if(pp.state->PreyY >= pp.world->height){
		pp.state->PreyY = pp.state->PreyY - pp.world->height;
//		pp.state->PreyY = pp.state->PreyY-1;
	}
	if(pp.state->PreyX < 0){
		pp.state->PreyX = pp.state->PreyX + pp.world->length;
//		pp.state->PreyX = pp.state->PreyX+1;
	}
	if(pp.state->PreyY < 0){
		pp.state->PreyY = pp.state->PreyY + pp.world->height;
//		pp.state->PreyY = pp.state->PreyY+1;
	}

}

State* getState(PredatorPrey pp){
	return pp.state;
}

Gridworld* getWorld(PredatorPrey pp){
	return pp.world;
}

bool Caught(PredatorPrey pp){
	return caught;
}



#endif

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




#endif

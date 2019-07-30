/*
 * feedforward.h
 *
 *  Created on: 15 Jul 2019
 *      Author: senpai
 */

#ifndef FEEDFORWARD_H_
#define FEEDFORWARD_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

//include other files
#include "population.h"
#include "neuron.h"
#include "sigmoid.h"

struct feedForward {
	int ID;
	double* Activation;
	Neuron* HiddenUnits;
	int NumInputs;
	int NumOutputs;
	bool bias;
	int Trials;
	int Fitness;
	int Catches;
	int Parent1;
	int Parent2;
	char* name;
	int GeneSize;
	int numHidden;
};

int counter = 0;
feedForward* newFeedForward(int in, int hid, int out, bool bias){
	counter++;
	int genesize = in + out;
	if(bias){
		genesize++;
	}
	double* actArr = new double[hid];
	Neuron* neuArr = new Neuron[hid];
	feedForward* ff = new feedForward{counter, actArr, neuArr, in, out, bias, 0, 0, 0, -1, -1, "Feed Forward", genesize, hid};
	return ff;
}

double* Activate(feedForward f, double* input, int inputLen, double* output){
	for(int key = 0; key< f.numHidden;key++){
		Neuron n = f.HiddenUnits[key];
		if(!n.Lesioned){
			for(int i = 0; i< inputLen;i++){
				f.Activation[key] = f.Activation[key] + (n.Weight[i] * input[i]);
			}
			f.Activation[key] = Logistic(1.0, f.Activation[key]);
		}
	}
	for(int i=0;i<f.NumOutputs;i++){
		for(int key = 0; key<f.numHidden; key++){
			Neuron n = f.HiddenUnits[key];
			output[i] = output[i] + (f.Activation[key] * n.Weight[inputLen + i]);
		}
//		output[i] = Direction((double)output[i]);//call different
	}
	return output;
}

Neuron* getHiddenUnits(feedForward f){
	return f.HiddenUnits;
}


void Create(feedForward f, Population* p, int numPops){
	for(int i = 0; i < numPops;i++){
		f.HiddenUnits[i] = selectNeuron(p[i]);
	}
}

int getTotalInputs(feedForward f){
	if(f.bias){
		return f.NumInputs +1;
	}else{
		return f.NumInputs;
	}
}

int getTotalOutputs(feedForward f){
	return f.NumOutputs;
}

bool hasBias(feedForward f){
	return f.bias;
}

void setFitness(feedForward f, int fitness){
	f.Fitness = fitness;
}

void setCatches(feedForward f,int c){
	f.Catches = c;
}

int getCatches(feedForward f){
	return f.Catches;
}

int getFitness(feedForward f){
	return f.Fitness;
}

int getID(feedForward f){
	return f.ID;
}

void setNeuronFitness(feedForward f){
	for(int i = 0; i<f.numHidden;i++){
		Neuron n = f.HiddenUnits[i];
		setFitness(n, f.Fitness);
		n.Trials++;
	}
}

void Tag(feedForward f){
	for(int i = 0; i<f.numHidden;i++){
		Neuron n = f.HiddenUnits[i];
		n.Tag = true;
	}
}

void resetActivation(feedForward f){
	f.Activation = new double[f.numHidden];
}

void resetFitness(feedForward f){
	f.Fitness = 0;
	f.Trials = 0;
}

void resetCatches(feedForward f){
	f.Catches = 0;
}
#endif

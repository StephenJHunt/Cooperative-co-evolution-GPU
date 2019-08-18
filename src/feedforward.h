/*
 * feedforward.h
 *
 *  Created on: 15 Jul 2019
 *      Author: senpai
 */

#ifndef FEEDFORWARD_H_
#define FEEDFORWARD_H_
#ifndef nPreds
#define nPreds 3
#endif
#ifndef nHidden
#define nHidden 15
#endif
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

struct aTeam{
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

struct feedForward {
	int ID;
//	double* Activation;
	double Activation[15];
//	Neuron* HiddenUnits;
	Neuron HiddenUnits[15];
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
//	double* actArr = new double[hid];
//	Neuron* neuArr = new Neuron[hid];
//	double actArr[15];
//	Neuron neuArr[15];
//	feedForward* ff = new feedForward{counter, actArr, neuArr, in, out, bias, 0, 0, 0, -1, -1, "Feed Forward", genesize, hid};
	feedForward* ff = new feedForward;
//	Neuron* n = new Neuron;
//	for(int i=0;i<hid;i++){
//		ff->Activation[i] = 0;
//		ff->HiddenUnits[i] = *n;
//	}
//	ff->Activation = actArr;
	ff->Catches=0;
	ff->Fitness=0;
	ff->GeneSize=genesize;
//	ff->HiddenUnits=neuArr;
	ff->ID=counter;
	ff->NumInputs=in;
	ff->NumOutputs=out;
	ff->Parent1=-1;
	ff->Parent2=-1;
	ff->Trials=0;
	ff->bias=false;
	ff->name="Feed Forward";
	ff->numHidden=hid;
	return ff;
}

void ff_reset(feedForward* ff,int in, int hid, int out, bool bias){
	ff->Catches=0;
	ff->Fitness=0;
	ff->GeneSize=in+out;
	ff->ID=counter;
	ff->NumInputs=in;
	ff->NumOutputs=out;
	ff->Parent1=-1;
	ff->Parent2=-1;
	ff->Trials=0;
	ff->bias=false;
	ff->name="Feed Forward";
	ff->numHidden=hid;
}

double* h_Activate(aTeam* h_t, double* h_input, int h_inputLen, double* h_output){
	for(int key = 0; key<h_t->numHidden;key++){
		for(int i=0;i<h_inputLen;i++){
			h_t->act1[key] += h_t->t1[key].Weight[i] * h_input[i];
			h_t->act2[key] += h_t->t2[key].Weight[i] * h_input[i];
			h_t->act3[key] += h_t->t3[key].Weight[i] * h_input[i];
		}
		h_t->act1[key] = h_Logistic(1.0, h_t->act1[key]);
		h_t->act2[key] = h_Logistic(1.0, h_t->act2[key]);
		h_t->act3[key] = h_Logistic(1.0, h_t->act3[key]);
	}
	for(int i=0;i<h_t->numOutputs;i++){
		for(int key = 0;key<h_t->numHidden;key++){
			h_output[i] += h_t->act1[key] * h_t->t1[key].Weight[h_inputLen +i];
			h_output[i] += h_t->act2[key] * h_t->t2[key].Weight[h_inputLen +i];
			h_output[i] += h_t->act3[key] * h_t->t3[key].Weight[h_inputLen +i];
		}
	}
	return h_output;
}

__device__ double* Activate(aTeam* t, double* input, int inputLen, double* output){
	for(int key = 0; key< t->numHidden;key++){
		Neuron n1 = t->t1[key];
		Neuron n2 = t->t2[key];
		Neuron n3 = t->t3[key];
		for(int i = 0; i< inputLen;i++){
			t->act1[key] = t->act1[key] + (n1.Weight[i] * input[i]);
			t->act2[key] = t->act2[key] + (n2.Weight[i] * input[i]);
			t->act3[key] = t->act3[key] + (n3.Weight[i] * input[i]);
		}
		t->act1[key] = Logistic(1.0, t->act1[key]);
		t->act2[key] = Logistic(1.0, t->act2[key]);
		t->act3[key] = Logistic(1.0, t->act3[key]);
	}
	for(int i=0;i<t->numOutputs;i++){
		for(int key = 0; key<t->numHidden; key++){
			Neuron n1 = t->t1[key];
			Neuron n2 = t->t2[key];
			Neuron n3 = t->t3[key];
			output[i] = output[i] + (t->act1[key] * n1.Weight[inputLen + i]);
			output[i] = output[i] + (t->act2[key] * n2.Weight[inputLen + i]);
			output[i] = output[i] + (t->act3[key] * n3.Weight[inputLen + i]);
		}
//		output[i] = Direction((double)output[i]);//call different
	}
	return output;
}

Neuron* getHiddenUnits(feedForward f){
	return f.HiddenUnits;
}


feedForward Create(feedForward f, Population* p, int numPops){
	for(int i = 0; i < numPops;i++){
		f.HiddenUnits[i] = selectNeuron(p[i]);
	}
	return f;
}

__device__ void kernelCreate(feedForward f, Population* p, int numPops){

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
//	f.Activation = new double[f.numHidden];
	for(int i =0;i<f.numHidden;i++){
		f.Activation[i] = 0;
	}
}

void resetFitness(feedForward f){
	f.Fitness = 0;
	f.Trials = 0;
}

void resetCatches(feedForward f){
	f.Catches = 0;
}
#endif

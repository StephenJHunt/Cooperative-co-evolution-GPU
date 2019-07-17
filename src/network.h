#ifndef NETWORK_H_
#define NETWORK_H_

#include "population.h"
#include "neuron.h"

class Network{
	public:
		double* Activate(double*, double*);
		void Create(Population* p);
		Neuron* getHiddenUnits();
		int getTotalInputs();
		int getTotalOutputs();
		bool hasBias();
		void setFitness(int);
		int getFitness();
		void setNeuronFitness();
		void resetActivation();
		void resetFitness();
		void Tag();
		int getID();
};



#endif

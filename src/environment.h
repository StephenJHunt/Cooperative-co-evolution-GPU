#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

struct Gridworld{
	int length;
	int height;
};

struct State{
	int* PredatorX;
	int* PredatorY;
	int PreyX;
	int PreyY;
};

class Environment{
public:
	Gridworld* getWorld();
	State* getState();
	bool Caught();
	bool Surrounded();
	void PerformPreyAction(int);
	void PerformPredatorAction(int);
	void Reset(int);
};



#endif

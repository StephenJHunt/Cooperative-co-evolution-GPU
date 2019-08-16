#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

struct Gridworld{
	int length;
	int height;
};

struct State{
	int PredatorX[3];
	int PredatorY[3];
	int PreyX;
	int PreyY;
	bool Caught;
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

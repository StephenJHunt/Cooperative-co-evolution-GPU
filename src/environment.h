#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_
#ifndef nPreds
#define nPreds 6
#endif
struct Gridworld{
	int length;
	int height;
};

struct State{
	int PredatorX[nPreds];
	int PredatorY[nPreds];
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

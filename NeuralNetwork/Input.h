#pragma once
#include"Layer.h"

class Input :
	public Layer
{
public:
	Input(int _NumNeurons) : Layer(_NumNeurons) {}

	void ForwardPropagation(vector<double> const& input) {
		A = input;
		NextLayer->ForwardPropagation(A);
	}

	void BackPropoagation(vector<double>& delta) {}
};
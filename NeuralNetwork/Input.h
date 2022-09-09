#pragma once
#include"Layer.h"

class Input :
	public Layer
{
public:
	Input(int _NumNeurons) : Layer(_NumNeurons) {}

	void ForwardPropagation(vector<double> const& input) {
		NextLayer->ForwardPropagation(input);
	}

	void BackPropoagation(vector<double>& delta) {}
};
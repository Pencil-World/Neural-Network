#pragma once
#include"Layer.h"

class Input :
	public Layer
{
public:
	using Layer::Layer;

	void ForwardPropagation(vector<double> const& input) {
		A = input;
		NextLayer->ForwardPropagation(A);
	}

	void BackPropoagation(vector<double>& delta) {}
};
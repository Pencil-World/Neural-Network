#pragma once
#include "Layer.h"

double sigmoid(double z) {
	return 1 / (1 + exp(z));
}

double sigmoid_prime(double z) {
	return sigmoid(z) * (1 - sigmoid(z));
}

class Output :
	public Layer
{
public:
	Output(int _NumFeatures) : Layer(_NumFeatures) {}

	void ForwardPropagation(vector<double> const& input) {
		transform(begin(W), end(W), begin(B), begin(Z), [&input](vector<double> const& w, double b) { return f(w, b, input); });
		transform(begin(Z), end(Z), begin(A), [](double z) { return sigmoid(z); });
	}

	void BackPropagation(double output, double target) {
		vector<double> error{ (output - target) * sigmoid_prime(Z[0]) };
		d_W = matmul(error, PrevLayer->A);
		d_B = error;
		PrevLayer->BackPropoagation(error);
	}
};
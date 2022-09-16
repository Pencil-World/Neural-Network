#pragma once
#include "Layer.h"

double sigmoid(double z) {
	return 1 / (1 + exp(-z));
}

double sigmoid_prime(double z) {
	return sigmoid(z) * (1 - sigmoid(z));
}

// MSE loss and logistic loss have the same derivative: output - target
class Output :
	public Layer
{
public:
	using Layer::Layer;

	void ForwardPropagation(vector<double> const& input) {
		transform(begin(W), end(W), begin(B), begin(Z), [&input](vector<double> const& w, double b) { return f(w, b, input); });
		transform(begin(Z), end(Z), begin(A), [](double z) { return sigmoid(z); });
	}

	void BackPropagation(double output, double target) {
		double activation_derivative(sigmoid_prime(Z[0]));
		vector<double> error{ (output - target) * activation_derivative};
		d_W = matmul(error, PrevLayer->A);
		d_B = error;

		PrevLayer->BackPropoagation(error);
	}
};
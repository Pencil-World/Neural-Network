#pragma once
#include"Input.h"
#include"Output.h"

class NN
{
private:
	Input input;
	Output output;
	vector<Layer> hidden;

	mt19937 rng;
public:
	// Model is a list of the number of nodes in each layer. Input layer (layer 0) is included. 
	NN(double alpha, double DropoutRate, vector<int> const& _model) : input(Input(_model.front())), output(Output(_model.back())) {
		int seed = random_device()();
		rng.seed(seed);

		function<bool()> lambda = [this, DropoutRate, prob(uniform_real_distribution<>(0, 1))]() { return prob(rng) < DropoutRate; };
		function<double(int, int)> kaiming_initialization([this](int fan_in, int fan_out) { normal_distribution<> dist(0, sqrt(2.0 / fan_in)); return dist(rng); });
		function<double(int, int)> xavier_initialization([this](int fan_in, int fan_out) { normal_distribution<> dist(0, sqrt(2.0 / (fan_in + fan_out))); return dist(rng); });

		transform(begin(_model) + 1, end(_model) - 1, back_inserter(hidden), [](int num) { return Layer(num); });
		Layer* prev = &input;
		for (auto& item : hidden) {
			item.assign(alpha, lambda, prev, kaiming_initialization);
			prev = &item;
		}

		output.assign(alpha, lambda, prev, xavier_initialization);
		cout << endl << "Constructing Neural Network\n\nSeed is: " << seed << endl;
	}

	double ForwardPropagation(vector<double> const& _input) {
		input.ForwardPropagation(_input);
		return output.A[0];
	}

	void BackPropagation(double _output, double _target) {
		output.BackPropagation(_output, _target);
	}

	void GradientDescent() {
		ranges::for_each(hidden, [](auto& val) { val.GradientDescent(); });
		output.GradientDescent();
	}
};
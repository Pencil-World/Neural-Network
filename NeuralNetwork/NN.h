#pragma once
#include"Input.h"
#include"Output.h"

class NN
{
private:
	Input input;
	Output output;
	vector<Layer> hidden;
	const int NumLayers;

	random_device rd;
	mt19937 rng;
	uniform_real_distribution<> probability;
public:
	// Model is a list of the number of nodes in each layer. Input layer (layer 0) is included. 
	NN(vector<int> const& _model) : rng(rd()), probability(0, 1), NumLayers(_model.size()), input(Input(_model.front())), output(Output(_model.back())) {
		cout << "\nconstructing NN\n";
		function<double(int, int)> kaiming_initialization([this](int fan_in, int fan_out) { normal_distribution<> dist(0, sqrt(2.0 / fan_in)); return dist(rng); });
		function<double(int, int)> xavier_initialization([this](int fan_in, int fan_out) { normal_distribution<> dist(0, sqrt(2.0 / (fan_in + fan_out))); return dist(rng); });

		transform(begin(_model) + 1, end(_model) - 1, back_inserter(hidden), [](int num) { return Layer(num); });
		Layer* prev = &input;
		for (auto& item : hidden) {
			item.assign(prev, kaiming_initialization);
			prev = &item;
		}

		output.assign(prev, xavier_initialization);
	}

	double ForwardPropagation(vector<double> const& _input) {
		cout << "\nstarting forward propagation\n";
		input.ForwardPropagation(_input);
		return output.A[0];
	}

	void BackPropagation(double _output, double _target) {
		cout << "\nstarting backward propagation\n";
		output.BackPropagation(_output, _target);
	}

	void GradientDescent(double alpha, double lambda) {
		ranges::for_each(hidden, [alpha, lambda](auto& val) { val.GradientDescent(alpha, lambda); });
	}
};
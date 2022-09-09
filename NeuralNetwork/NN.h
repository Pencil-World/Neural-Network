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
public:
	// Model is a list of the number of nodes in each layer. Input layer (layer 0) is included. 
	NN(vector<int> const& _model) : NumLayers(_model.size()), input(Input(_model.front())), output(Output(_model.back())) {
		transform(begin(_model) + 1, end(_model) - 1, back_inserter(hidden), [](int num) { return Layer(num); });
		Layer* prev = &input;
		for (auto& item : hidden) {
			item.assign(prev);
			prev = &item;
		}

		output.assign(prev);
	}

	double ForwardPropagation(vector<double> const& _input) {
		input.ForwardPropagation(_input);
		return output.A[0];
	}

	void BackPropagation(double _output, double _target) {
		output.BackPropagation(_output, _target);
	}

	void GradientDescent(double alpha, double lambda) {
		ranges::for_each(hidden, [alpha, lambda](auto& val) { val.GradientDescent(alpha, lambda); });
	}
};
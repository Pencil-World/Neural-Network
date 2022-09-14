#pragma once
#include<algorithm>
#include<chrono>
#include<cmath>
#include<fstream>
#include<functional>
#include<iostream>
#include<numeric>
#include<numbers>
#include<random>
#include<string>
#include<vector>
#include"Helper.h"

using namespace std;

// predicted value using dot product
double f(vector<double> const& w, double b, vector<double> const& x) {
	// weights and biases DIRECTLY control the output of the function. why linear? because linear is like an atom, it is the building block of all functions (plus relu)
	return inner_product(begin(w), end(w), begin(x), b);
}

// activation function. relu function. prevents final model from being linear regression. linear algebra: a linear function of a linear function is itself a linear function
double relu(double z) {
	return max(0.0, z);
}

double relu_prime(double z) {
	return !signbit(z);
}

void print(vector<double> const& temp) {
	ranges::for_each(temp, [](double val) { cout << '\t' << val; });
	cout << endl;
}

// capital variable names denote they are 1 dimensional higher than their previous counterpart often indicating a matrix or vector
class Layer {
	friend class NN;
	friend class Input;
	friend class Output;
private:
	int NumNeurons{ 0 }, NumFeatures{ 0 };
	Layer* PrevLayer{ nullptr };
	Layer* NextLayer{ nullptr };
	vector<vector<double>> W, d_W;
	vector<double> B, Z, A, d_B;

	void reset() {
		d_W = vector<vector<double>>(NumNeurons, vector<double>(NumFeatures));
		d_B = vector<double>(NumNeurons);
	}

	void printForwardPropagation() {
		cout << "\nstarting forward propagation of " << NumNeurons << " neuron layer\n";
		vector<vector<double>> W_T(transpose(W));
		ranges::for_each(W_T, [this](vector<double> const& temp) { cout << "W matrix: "; print(temp); });
		cout << "B matrix: ";
		print(B);
		cout << "Z matrix: ";
		print(Z);
		cout << "A matrix: ";
		print(A);
	}

	void printBackPropagation() {
		cout << "\nstarting backward propagation of " << NumNeurons << " neuron layer\n";
		vector<vector<double>> d_W_T(transpose(d_W));
		ranges::for_each(d_W_T, [this](vector<double> const& temp) { cout << "d_W matrix: "; print(temp); });
		cout << "d_B matrix: ";
		print(d_B);
	}
public:
	// NumWeights comes before NumNeurons because NumWeights corresponds to the # of inputs, x, while NumNeurons corresponds to the # of outputs, activation
	Layer(int _NumNeurons) : NumNeurons(_NumNeurons) {}

	//random weight generation prevents all nodes from a given layer from being identical
	void assign(Layer* _PrevLayer, function<double(int, int)> _initialization) {
		cout << NumNeurons << " neuron layer PrevLayer = " << _PrevLayer->NumNeurons << " neuron layer\n";
		PrevLayer = _PrevLayer;
		PrevLayer->NextLayer = this;
		NumFeatures = PrevLayer->NumNeurons;

		generate_n(back_inserter(W), NumNeurons, [=]() { vector<double> temp; generate_n(back_inserter(temp), NumFeatures, [=]() { return _initialization(NumFeatures, NumNeurons); }); return temp; });
		B = Z = A = vector<double>(NumNeurons);
		reset();
	}

	virtual void ForwardPropagation(vector<double> const& input) {
		transform(begin(W), end(W), begin(B), begin(Z), [&input](vector<double> const& w, double b) { return f(w, b, input); });
		transform(begin(Z), end(Z), begin(A), [](double z) { return relu(z); });

		printForwardPropagation();
		NextLayer->ForwardPropagation(A);
	}

	// backpropagate the error. Gradient descent changes the weights to reduce the error
	virtual void BackPropoagation(vector<double>& delta) {
		vector<double> activation_derivative;
		ranges::transform(Z, back_inserter(activation_derivative), [](double val) { return relu_prime(val); });
		delta = matmul(transpose(NextLayer->W), delta) * activation_derivative;
		d_W = d_W + matmul(delta, PrevLayer->A);
		d_B = d_B + delta;

		printBackPropagation();
		cout << "A_der matrix: ";
		print(activation_derivative);
		cout << "delta matrix: ";
		print(delta);

		PrevLayer->BackPropoagation(delta);
	}

	void GradientDescent(double alpha, double lambda) {
		W = W - alpha * d_W;
		B = B - alpha * d_B;
		reset();
	}
};
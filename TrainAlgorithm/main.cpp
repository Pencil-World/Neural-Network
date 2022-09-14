#include<algorithm>
#include<chrono>
#include<cmath>
#include<fstream>
#include<functional>
#include<iostream>
#include<numeric>
#include<numbers>
#include<string>
#include<tuple>
#include<vector>
#include <iomanip>

#include"NN.h"
#include"Helper.h"

#pragma warning( once : 4267 )
#pragma warning( once : 4244 )
using namespace std;

int DataLength, DataWidth;
double alpha, lambda; // alpha accounts for learning rate and low alpha reduces underfitting. lambda accounts for regularization and high lambda reduces overfitting

// logistic regression calculates probability for 2 classes (1 or 0). accepts multiple inputs and computes them into different features (x => x^2, x, log x)
// this example uses f(x) = x1^2 + x1 + x2^3 + x2 + b

double loss(double output, double target) {
	return pow(output - target, 2);
}

//logistic loss

// transform parameters are arranged in anti-ocd order. pushes weights and biases towards to lowest cost by comparing derivative and calculating the steepest slope
double Train(NN& nn, vector<pair<vector<double>, int>>& train_set) {
	double cost = 0;
	cout << "\nstart training\n";
	for (auto const& coord : train_set) {
		double output = nn.ForwardPropagation(coord.first);
		double target = coord.second;
		nn.BackPropagation(output, target);
		cost += loss(output, target);
	}

	nn.GradientDescent(alpha, lambda);
	return cost / DataLength;
}

void LoadData(vector<pair<vector<double>, int>>& data) {
	cout << "\nLoadData\n";
	fstream file("C://Users//iayfn//source//repos//Neural Network//NeuralNetwork//data.txt");
	int output;
	file >> DataWidth;
	vector<double> input(DataWidth);
	while (file >> output) {
		for (int i = 0; i < DataWidth; ++i, file >> output)
			input[i] = output;
		data.push_back({ input, output });
	}

	file.close();
}

// feature engineering & feature selection to prevent underfitting
void ZScoreNormalization(vector<pair<vector<double>, int>>& data) {
	cout << "\nZScoreNormalization\n";
	vector<double> mean, variance, std;
	mean = (1.0 / DataLength) * accumulate(begin(data), end(data), vector<double>(DataWidth, 0), [](auto const& total, auto const& coord) { return total + coord.first; });
	variance = (1.0 / DataLength) * accumulate(begin(data), end(data), vector<double>(DataWidth, 0), [&mean](auto const& total, auto const& coord) { auto val = coord.first - mean; return total + val * val; });
	ranges::transform(variance, back_inserter(std), [](double val) { return sqrt(val); });
	ranges::for_each(data, [&mean, &std](auto& coord) { coord.first = (coord.first - mean) / std; });
}

int main() {
	cout << fixed << setprecision(4) << "\nstart program\n";
	const auto start = chrono::system_clock::now();
	vector<pair<vector<double>, int>> data;
	LoadData(data);
	DataLength = data.size();
	ZScoreNormalization(data);

	cout << "\nsplit data\n";
	// split. train = 60%, cross_validation = 20%, test = 20%
	vector<pair<vector<double>, int>> train_set =				vector<pair<vector<double>, int>>(begin(data) + DataLength * 0.0, begin(data) + DataLength * 0.6);
	vector<pair<vector<double>, int>> cross_validation_set =	vector<pair<vector<double>, int>>(begin(data) + DataLength * 0.6, begin(data) + DataLength * 0.8);
	vector<pair<vector<double>, int>> test_set =				vector<pair<vector<double>, int>>(begin(data) + DataLength * 0.8, begin(data) + DataLength * 1.0);

	DataLength = train_set.size();
	const double LearningRate = 0.01; // start between 0.1 - 0.01. 
	alpha = LearningRate / DataLength;
	// lambda = 

	NN nn({ DataWidth, 5, 3, 1 });
	// cost approaches an asymptote, never reaching zero. each iteration reduces cost because linear regression & mean squared error create a convex cost function
	for (int it = 0; ; ++it) {
		double cost = Train(nn, train_set);
		if (it % 1000 == 0) {
			const std::chrono::duration<double> diff = chrono::system_clock::now() - start;
			cout << endl << diff.count() << '\t' << cost << endl;
		}
	}

	const std::chrono::duration<double> diff = chrono::system_clock::now() - start;
	cout << diff.count();
	return 0;
}
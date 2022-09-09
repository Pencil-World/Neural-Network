#pragma once
#include<algorithm>
#include<iterator>
#include<numeric>
#include<functional>
#include<vector>

using namespace std;


template<class T>
vector<T> operator+(vector<T> const& lhs, vector<T> rhs) {
	ranges::transform(lhs, rhs, begin(rhs), [](T _lhs, T _rhs) { return _lhs + _rhs; });
	return rhs;
}

template<class T>
vector<T> operator-(vector<T> const& lhs, vector<T> rhs) {
	ranges::transform(lhs, rhs, begin(rhs), [](T _lhs, T _rhs) { return _lhs - _rhs; });
	return rhs;
}

template<class T>
vector<T> operator*(T lhs, vector<T> rhs) {
	ranges::for_each(rhs, [lhs](T& _rhs) { _rhs *= lhs; });
	return rhs;
}

template<class T>
vector<vector<T>> operator*(T lhs, vector<vector<T>> rhs) {
	ranges::for_each(rhs, [lhs](vector<T>& val) { val = lhs * val; });
	return rhs;
}

template<class T>
vector<T> operator*(vector<T> const& lhs, vector<T> rhs) {
	ranges::transform(lhs, rhs, begin(rhs), multiplies<T>());
	return rhs;
}

template<class T>
vector<T> operator/(vector<T> const& lhs, vector<T> rhs) {
	ranges::transform(lhs, rhs, begin(rhs), divides<T>());
	return rhs;
}

/*
All generic operation functions are specialized for NN purposes
matrix multiplication or numpy.matmul()
schur product or hadamard product
outer product or tensor product
*/

// matrix multiplication using inner_product(). lhs expects a transposed weight matrix and rhs expects a transposed delta (column -> row vector). lhs[0].size() == rhs.size()
vector<double> matmul(vector<vector<double>> const& lhs, vector<double> const& rhs) {
	vector<double> product;
	ranges::transform(lhs, back_inserter(product), [&rhs](vector<double> const& val) { return inner_product(begin(val), end(val), begin(rhs), 0.0); });
	return product;
}

// outer product is derived from matrix multiplication. lhs expects a column vector and rhs expects a row vector
vector<vector<double>> matmul(vector<double> const& lhs, vector<double> const& rhs) {
	vector<vector<double>> product;
	ranges::transform(lhs, back_inserter(product), [&rhs](double val) { return val * rhs; });
	return product;
}

// only for weight matrix. all other member arrays in Layer are 1D vectors
vector<vector<double>> transpose(vector<vector<double>> const& matrix) {
	int row(matrix[0].size()), col(matrix.size());
	vector<vector<double>> temp(row, vector<double>(col));
	while (--row >= 0)
		for (int num = col; --num >= 0;)
			temp[row][col] = matrix[col][row];
	return temp;
}
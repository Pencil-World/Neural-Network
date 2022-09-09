#include<array>
#include<chrono>
#include<cmath>
#include<fstream>
#include<iostream>
#include<random>

using namespace std;

bool f(array<double, 2> const& in) {
    return pow(in[0] - 40, 2) + pow(in[1] - 60, 2) < 1225;//35^2
}

int main() {
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<> dist(0, 100);
    const string path("C://Users//iayfn//source//repos//Neural Network//NeuralNetwork//data.txt");

    fstream file;
    array<double, 2> in;
    const auto start = chrono::system_clock::now();
    for (int count = 0; ; ++count) {
        if (count % 1000 == 0) {
            file.close();
            const std::chrono::duration<double> diff = chrono::system_clock::now() - start;
            cout << diff.count() << '\t' << count << endl;
            file.open(path);
            file.seekp(0, ios_base::end);
        }

        generate(begin(in), end(in), [&rng, &dist, &file]() { double num = dist(rng); file << num << " "; return num; });
        file << f(in) << endl;
    }

    return 0;
}
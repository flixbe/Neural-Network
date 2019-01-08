#pragma once

#include <vector>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numberOutputs, unsigned index);
	double getOutputValue() const { return outputValue; }
	void setOutputValue(double value) { outputValue = value; }
	void feedForward(const Layer &prevLayer);
	void calculateOutputGradients(double targetValues);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &previousLayer);
private:
	std::vector<Connection> outputWeights;
	unsigned m_Index;
	double outputValue;
	double gradient;
	static double eta;
	static double alpha;
	static double randomWeight() { return rand() / double(RAND_MAX); }
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);
	double sumDOW(const Layer &nextLayer) const;
};
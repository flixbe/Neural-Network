#include "Neuron.h"

double Neuron::eta   = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numberOutputs, unsigned index)
{
	for (unsigned c = 0; c < numberOutputs; ++c)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	m_Index = index;
}

void Neuron::feedForward(const Layer &previousLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < previousLayer.size(); ++n)
	{
		sum += previousLayer[n].getOutputValue() * previousLayer[n].outputWeights[m_Index].weight;
	}
	outputValue = Neuron::activationFunction(sum);
}

double Neuron::activationFunction(double x)
{
	return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activationFunctionDerivative(outputValue);
}

void Neuron::calculateOutputGradients(double targetValues)
{
	double delta = targetValues - outputValue;
	gradient = delta * Neuron::activationFunctionDerivative(outputValue);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer &previousLayer)
{
	for (unsigned n = 0; n < previousLayer.size(); ++n)
	{
		Neuron &neuron = previousLayer[n];
		double oldDeltaWeight = neuron.outputWeights[m_Index].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputValue() * gradient + alpha * oldDeltaWeight;
		neuron.outputWeights[m_Index].deltaWeight = newDeltaWeight;
		neuron.outputWeights[m_Index].weight += newDeltaWeight;
	}
}
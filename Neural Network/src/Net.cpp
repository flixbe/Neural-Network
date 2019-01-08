#include "Training.h"
#include "Neuron.h"
#include "Net.h"

double Net::recentAverageSmoothingFactor = 100.0;

Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numberLayers = topology.size();
	for (unsigned layerNumber = 0; layerNumber < numberLayers; ++layerNumber)
	{
		layers.push_back(Layer());
		unsigned numberOutputs = layerNumber == topology.size() - 1 ? 0 : topology[layerNumber + 1];

		for (unsigned neuronNumber = 0; neuronNumber <= topology[layerNumber]; ++neuronNumber)
			layers.back().push_back(Neuron(numberOutputs, neuronNumber));

		layers.back().back().setOutputValue(1.0);
	}
}

void Net::feedForward(const std::vector<double> &inputValues)
{
	assert(inputValues.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < inputValues.size(); i++)
	{
		layers[0][i].setOutputValue(inputValues[i]);
	}

	for (unsigned layerNumber = 1; layerNumber < layers.size(); ++layerNumber)
	{
		Layer &previousLayer = layers[layerNumber - 1];
		for (unsigned n = 0; n < layers[layerNumber].size() - 1; ++n)
		{
			layers[layerNumber][n].feedForward(previousLayer);
		}
	}
}

void Net::backPropagation(const std::vector<double> &targetValues)
{
	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		error += delta * delta;
	}

	error /= outputLayer.size() - 1;
	error = sqrt(error);

	recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calculateOutputGradients(targetValues[n]);
	}
	
	for (unsigned layerNumber = layers.size() - 2; layerNumber > 0; --layerNumber)
	{
		Layer &hiddenLayer = layers[layerNumber];
		Layer &nextLayer = layers[layerNumber + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	for (unsigned layerNumber = layers.size() - 1; layerNumber > 0; --layerNumber)
	{
		Layer &layer = layers[layerNumber];
		Layer &previousLayer = layers[layerNumber - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(previousLayer);
		}
	}
}

void Net::getResults(std::vector<double> &resultValues) const
{
	resultValues.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		resultValues.push_back(layers.back()[n].getOutputValue());
	}
}
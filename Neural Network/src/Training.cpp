#include "Training.h"

Training::Training(const std::string filename)
{
	trainingDataSetFile.open(filename.c_str());
}

void Training::getTopology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	std::getline(trainingDataSetFile, line);
	std::stringstream ss(line);
	ss >> label;

	if (this->isEOF() || label.compare("Topology:") != 0)
		abort();

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	
	return;
}

unsigned Training::getNextInputs(std::vector<double> &inputValues)
{
	inputValues.clear();

	std::string line;
	std::getline(trainingDataSetFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("Input:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
			inputValues.push_back(oneValue);
	}

	return inputValues.size();
}

unsigned Training::getTargetOutputs(std::vector<double> &targetOuputValues)
{
	targetOuputValues.clear();

	std::string line;
	std::getline(trainingDataSetFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("Output:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
			targetOuputValues.push_back(oneValue);
	}

	return targetOuputValues.size();
}
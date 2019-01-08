#include "Training.h"
#include "Neuron.h"
#include "Net.h"

void showVectorValues(std::string label, std::vector<double> vector)
{
	std::cout << label << " ";

	for (unsigned i = 0; i < vector.size(); i++)
	{
		std::cout << vector[i] << " ";
	}

	std::cout << std::endl;
}

int main()
{
	Training training("data.txt");
	std::vector<unsigned> topology;
	training.getTopology(topology);
	Net net(topology);

	std::vector<double> inputValues, targetValues, resultValues;
	int trainingPass = 0;

	while (!training.isEOF())
	{
		++trainingPass;
		std::cout << std::endl << "Pass: " << trainingPass << std::endl;

		if (training.getNextInputs(inputValues) != topology[0])
			break;

		showVectorValues("Input: ", inputValues);
		net.feedForward(inputValues);

		training.getTargetOutputs(targetValues);
		showVectorValues("Targets: ", targetValues);
		assert(targetValues.size() == topology.back());

		net.getResults(resultValues);
		showVectorValues("Outputs: ", resultValues);

		net.backPropagation(targetValues);

		std::cout << "Net average error: " << net.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done!" << std::endl;

	system("PAUSE");

	return 0;
}
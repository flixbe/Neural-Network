#pragma once

#include <vector>

class Net
{
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputValues);
	void backPropagation(const std::vector<double> &targetValues);
	void getResults(std::vector<double> &resultValues) const;
	double getRecentAverageError() const { return recentAverageError; }
private:
	std::vector<Layer> layers;
	double error;
	double recentAverageError;
	static double recentAverageSmoothingFactor;
};
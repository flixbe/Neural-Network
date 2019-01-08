#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class Training
{
public:
	Training(const std::string filename);
	bool isEOF() { return trainingDataSetFile.eof(); }
	void getTopology(std::vector<unsigned> &topology);
	unsigned getNextInputs(std::vector<double> &inputValues);
	unsigned getTargetOutputs(std::vector<double> &targetOuputValues);
private:
	std::ifstream trainingDataSetFile;
};
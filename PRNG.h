#pragma once
#include <random>

class PRNG {
private:
    std::mt19937 mt;
    std::normal_distribution<double> dist;

public:
    PRNG(unsigned long seed); //Constructor to set up the seed once

    double getStandardNormal(); //Function to get the next random number
};
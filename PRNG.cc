#include "PRNG.h"
#include <random>

PRNG::PRNG(unsigned long seed) : mt(seed), dist(0.0, 1.0) {}

//This just generates and returns the next number
double PRNG::getStandardNormal() {
    return dist(mt);
}
#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "vector/vector.h"
#include <random>

class Random {
private:
    std::mt19937 gen;

public:

    Random(unsigned int seed = 42);

    float uniform(float min, float max);
    float normal(float mean, float stddev);

    Vector random_vector(size_t size, float min, float max);
    Vector mutate_vector(const Vector& genome, float rate, float scale);
};

#endif
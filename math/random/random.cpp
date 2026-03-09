#include "random/random.h"

Random::Random(unsigned int seed) : gen(seed) {}

float Random::uniform(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

float Random::normal(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(gen);
}

Vector Random::random_vector(size_t size, float min, float max) {
    Vector result(size);
    for (size_t i = 0; i < size; ++i) {
        result.at(i) = uniform(min, max);
    }
    return result;
}

Vector Random::mutate_vector(const Vector& genome, float rate, float scale) {
    Vector result(genome.size());
    std::uniform_real_distribution<float> chance(0, 1);
    std::uniform_real_distribution<float> mutation(-scale, scale);
    
    for (size_t i = 0; i < genome.size(); ++i) {
        float value = genome.at(i);
        if (chance(gen) < rate) {
            value += mutation(gen);
        }
        result.at(i) = value;
    }
    return result;
}
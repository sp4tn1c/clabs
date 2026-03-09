#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include "random/random.h"
#include <string>
#include <vector>

struct DataPoint {
    float x, y;
    int label;
};

class DataGenerator {
private:
    Random rng;
    
public:
    DataGenerator(unsigned int seed = 42);
    std::vector<DataPoint> generate(size_t n_per_class, float separation = 2.0f);
    void save_to_csv(const std::vector<DataPoint>& data, const std::string& filename);
    std::vector<DataPoint> load_from_csv(const std::string& filename);
};

#endif // GENERATOR_HPP
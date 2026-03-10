#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include "vector/vector.h"
#include "generator/generator.h"
#include <vector>

class Classifier {
private:
    Vector weights;  // [w1, w2, b]
    
public:
    Classifier(size_t input_size = 2);
    void set_weights(const Vector& w);
    Vector get_weights() const;
    int predict(const DataPoint& point) const;
    float accuracy(const std::vector<DataPoint>& data) const;
};

#endif // CLASSIFIER_HPP
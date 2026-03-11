#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>

class Vector {
private:
    std::vector<float> data;

public:
    Vector(size_t size);
    Vector(const std::vector<float>& data);

    float& at(size_t index);
    float at(size_t index) const;

    size_t size() const;

    float dot(const Vector& other) const;
    Vector add(const Vector& other) const;
    Vector multiply(float scalar) const;
};

#endif
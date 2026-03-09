#include "vector/vector.h"

Vector::Vector(size_t size) {
    data.resize(size, 0.0f);
}

Vector::Vector(const std::vector<float>& data) : data(data) {}

float& Vector::at(size_t index) {
    if (index >= data.size()) {
        throw std::out_of_range("Vector index out of bounds");
    }
    return data[index];
}

float Vector::at(size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Vector index out of bounds");
    }
    return data[index];
}

size_t Vector::size() const {
    return data.size();
}

float Vector::dot(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::runtime_error("Vector sizes don't match for dot product");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i] * other.data[i];
    }
    return sum;
}

Vector Vector::add(const Vector& other) const {
    if (data.size() != other.data.size()) {
        throw std::runtime_error("Vector sizes don't match for addition");
    }
    
    Vector result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Vector Vector::multiply(float scalar) const {
    Vector result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}
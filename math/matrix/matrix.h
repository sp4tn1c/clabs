#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "vector/vector.h"
#include <vector>
#include <stdexcept>

class Matrix {
private:
    std::vector<std::vector<float>> data;
    size_t rows_count;
    size_t cols_count;

public:
    Matrix(size_t rows, size_t cols);
    Matrix(const std::vector<std::vector<float>>& data);

    float& at(size_t row, size_t col);
    float at(size_t row, size_t col) const;

    size_t rows() const;
    size_t cols() const;

    Vector multiply(const Vector& vec) const;
    Matrix multiply(const Matrix& other) const;
    Matrix transpose() const;
};

#endif
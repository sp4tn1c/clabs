#include "matrix/matrix.h"

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_count(rows), cols_count(cols) {
    data.resize(rows, std::vector<float>(cols, 0.0f));
}

Matrix::Matrix(const std::vector<std::vector<float>>& data) 
    : data(data), 
      rows_count(data.size()), 
      cols_count(data.empty() ? 0 : data[0].size()) {}

float& Matrix::at(size_t row, size_t col) {
    if (row >= rows_count || col >= cols_count) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

float Matrix::at(size_t row, size_t col) const {
    if (row >= rows_count || col >= cols_count) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

size_t Matrix::rows() const {
    return rows_count;
}

size_t Matrix::cols() const {
    return cols_count;
}

Vector Matrix::multiply(const Vector& vec) const {
    if (cols_count != vec.size()) {
        throw std::runtime_error("Matrix cols != Vector size");
    }
    
    Vector result(rows_count);
    for (size_t i = 0; i < rows_count; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols_count; ++j) {
            sum += data[i][j] * vec.at(j);
        }
        result.at(i) = sum;
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols_count != other.rows_count) {
        throw std::runtime_error("Matrix A cols != Matrix B rows");
    }
    
    Matrix result(rows_count, other.cols_count);
    for (size_t i = 0; i < rows_count; ++i) {
        for (size_t j = 0; j < other.cols_count; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_count; ++k) {
                sum += data[i][k] * other.data[k][j];
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_count, rows_count);
    for (size_t i = 0; i < rows_count; ++i) {
        for (size_t j = 0; j < cols_count; ++j) {
            result.at(j, i) = data[i][j];  // ← Исправлено здесь
        }
    }
    return result;
}
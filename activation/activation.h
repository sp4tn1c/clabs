#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "vector/vector.h"

float sigmoid(float x);

Vector sigmoid(const Vector& v);

float relu(float x);
Vector relu(const Vector& v);

#endif
#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "../math/vector/vector.h"
#include "../math/matrix/matrix.h"
#include "../generator/generator.h"
#include <vector>

class NeuralNetwork {
private:
    int input_size, hidden_size, output_size;
    Matrix weights_input_hidden;   // [hidden][input]
    Vector bias_hidden;            // [hidden]
    Matrix weights_hidden_output;  // [output][hidden]
    Vector bias_output;            // [output]
    Vector hidden_layer_values;
    Vector output_layer_values;

    static float sigmoid(float x);

public:
    NeuralNetwork(int in_size, int hid_size, int out_size);

    void init_weights();

    // ← ДОБАВЛЕНО: const для вызова на константном объекте
    float predict(const std::vector<float>& features) const;

    // Для совместимости
    float predict(float x, float y) const { return predict({x, y}); }

    Vector get_weights() const;
    void set_weights(const Vector& weights);

    // ← ДОБАВЛЕНО: геттеры для доступа из NeuralTrainer
    int get_input_size() const { return input_size; }
    int get_hidden_size() const { return hidden_size; }
    int get_output_size() const { return output_size; }

    int getTotalWeightsCount() const;
};

#endif
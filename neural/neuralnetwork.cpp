#include "neural/neuralnetwork.h"
#include "../math/random/random.h"
#include <cmath>

float NeuralNetwork::sigmoid(float x) {
    return (x >= 0) ? 1.0f / (1.0f + std::exp(-x))
                    : std::exp(x) / (1.0f + std::exp(x));
}

NeuralNetwork::NeuralNetwork(int in_size, int hid_size, int out_size)
    : input_size(in_size), hidden_size(hid_size), output_size(out_size),
      weights_input_hidden(hid_size, in_size),
      bias_hidden(hid_size),
      weights_hidden_output(out_size, hid_size),
      bias_output(out_size),
      hidden_layer_values(hid_size),
      output_layer_values(out_size) {}

void NeuralNetwork::init_weights() {
    Random rng(42);
    for (int h = 0; h < hidden_size; ++h) {
        for (int i = 0; i < input_size; ++i)
            weights_input_hidden.at(h, i) = rng.normal(0.0f, 0.5f);
        bias_hidden.at(h) = 0.0f;
    }
    for (int o = 0; o < output_size; ++o) {
        for (int h = 0; h < hidden_size; ++h)
            weights_hidden_output.at(o, h) = rng.normal(0.0f, 0.5f);
        bias_output.at(o) = 0.0f;
    }
}

// ← ДОБАВЛЕНО: const в конце
float NeuralNetwork::predict(const std::vector<float>& features) const {
    // Локальные копии значений (т.к. метод const)
    std::vector<float> hidden_vals(hidden_size);
    std::vector<float> output_vals(output_size);

    for (int h = 0; h < hidden_size; ++h) {
        float sum = bias_hidden.at(h);
        for (int i = 0; i < input_size; ++i)
            sum += features[i] * weights_input_hidden.at(h, i);
        hidden_vals[h] = sigmoid(sum);
    }
    for (int o = 0; o < output_size; ++o) {
        float sum = bias_output.at(o);
        for (int h = 0; h < hidden_size; ++h)
            sum += hidden_vals[h] * weights_hidden_output.at(o, h);
        output_vals[o] = sigmoid(sum);
    }
    return output_vals[0];
}

Vector NeuralNetwork::get_weights() const {
    Vector w(hidden_size*input_size + hidden_size + output_size*hidden_size + output_size);
    int idx = 0;
    for (int h = 0; h < hidden_size; ++h)
        for (int i = 0; i < input_size; ++i) w.at(idx++) = weights_input_hidden.at(h, i);
    for (int h = 0; h < hidden_size; ++h) w.at(idx++) = bias_hidden.at(h);
    for (int o = 0; o < output_size; ++o)
        for (int h = 0; h < hidden_size; ++h) w.at(idx++) = weights_hidden_output.at(o, h);
    for (int o = 0; o < output_size; ++o) w.at(idx++) = bias_output.at(o);
    return w;
}

void NeuralNetwork::set_weights(const Vector& weights) {
    int idx = 0;
    for (int h = 0; h < hidden_size; ++h)
        for (int i = 0; i < input_size; ++i) weights_input_hidden.at(h, i) = weights.at(idx++);
    for (int h = 0; h < hidden_size; ++h) bias_hidden.at(h) = weights.at(idx++);
    for (int o = 0; o < output_size; ++o)
        for (int h = 0; h < hidden_size; ++h) weights_hidden_output.at(o, h) = weights.at(idx++);
    for (int o = 0; o < output_size; ++o) bias_output.at(o) = weights.at(idx++);
}

int NeuralNetwork::getTotalWeightsCount() const {
    return hidden_size*input_size + hidden_size + output_size*hidden_size + output_size;
}
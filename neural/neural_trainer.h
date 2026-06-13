#ifndef NEURAL_TRAINER_HPP
#define NEURAL_TRAINER_HPP

#include "neuralnetwork.h"
#include "../generator/generator.h"
#include <vector>

class NeuralTrainer {
private:
    NeuralNetwork& network;
    const std::vector<DataPoint>& train_data;
    int pop_size;
    int generations;
    float mut_rate;
    float mut_scale;

public:
    // ← ИСПРАВЛЕНО: все параметры
    NeuralTrainer(NeuralNetwork& net,
                  const std::vector<DataPoint>& data,
                  int pop = 50,
                  int gens = 200,
                  float rate = 0.2f,
                  float scale = 0.2f)
        : network(net),
          train_data(data),
          pop_size(pop),
          generations(gens),
          mut_rate(rate),
          mut_scale(scale) {}

    void train();
};

#endif
#include "neural_trainer.h"
#include "../math/random/random.h"
#include <iostream>

void NeuralTrainer::train() {
    Random rng(42);

    // ← ИСПРАВЛЕНО: используем геттеры вместо приватных полей
    int in_size = network.get_input_size();
    int hid_size = network.get_hidden_size();
    int out_size = network.get_output_size();

    Vector base = network.get_weights();
    std::vector<Vector> population(pop_size, base);

    std::cout << "  Training Neural Network..." << std::endl;

    for (int gen = 0; gen < generations; ++gen) {
        float best_acc = 0;
        Vector best_w = base;

        for (auto& ind : population) {
            ind = rng.mutate_vector(base, mut_rate, mut_scale);

            // ← ИСПРАВЛЕНО: создаём временную сеть с правильными размерами
            NeuralNetwork temp(in_size, hid_size, out_size);
            temp.set_weights(ind);

            // Простая оценка на части данных
            int correct = 0;
            int eval_count = std::min(50, static_cast<int>(train_data.size()));
            for (int i = 0; i < eval_count; ++i) {
                const auto& p = train_data[i];
                float pred = temp.predict(p.features);
                int pred_class = (pred >= 0.5f) ? 1 : 0;
                if (pred_class == p.label) correct++;
            }
            float acc = static_cast<float>(correct) / eval_count;

            if (acc > best_acc) { best_acc = acc; best_w = ind; }
        }

        if (gen % 50 == 0)
            std::cout << "  Gen " << gen << ": Acc ≈ " << (best_acc*100) << "%" << std::endl;

        base = best_w;
    }

    network.set_weights(base);
    std::cout << "  Training complete!" << std::endl;
}
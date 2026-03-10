#include "generator/generator.h"
#include "classifier/classifier.h"
#include "geneticalg/geneticalg.h"
#include <iostream>
#include <fstream>

int main() {
    std::cout << "=== LABA 1: Binary Classifier with GA ===" << std::endl;

    // 1. Генерация данных
    std::cout << "\n=== Step 1: Data Generation ===" << std::endl;
    DataGenerator gen(42);
    auto data = gen.generate(100, 2.0f);
    gen.save_to_csv(data, "data/train.csv");

    // 2. Обучение через ГА
    std::cout << "\n=== Step 2: Training with GA ===" << std::endl;
    GeneticAlg ga(42, 50, 100, 0.1f, 0.1f);
    Vector best_weights = ga.train(data);

    // 3. Классификатор с лучшими весами
    std::cout << "\n=== Step 3: Final Classification ===" << std::endl;
    Classifier clf(2);
    clf.set_weights(best_weights);

    std::cout << "\nBest weights:" << std::endl;
    std::cout << "  w1 = " << best_weights.at(0) << std::endl;
    std::cout << "  w2 = " << best_weights.at(1) << std::endl;
    std::cout << "  b  = " << best_weights.at(2) << std::endl;
    std::cout << "\nFinal accuracy: " << clf.accuracy(data) * 100 << "%" << std::endl;

    // 4. Экспорт результатов
    std::cout << "\n=== Step 4: Export Results ===" << std::endl;
    std::ofstream results("data/results.csv");
    results << "x,y,true,pred\n";
    for (const auto& p : data) {
        results << p.x << "," << p.y << "," << p.label << "," << clf.predict(p) << "\n";
    }
    results.close();

    std::ofstream weights_file("data/weights.csv");
    weights_file << "w1,w2,b\n" << best_weights.at(0) << "," << best_weights.at(1) << "," << best_weights.at(2) << "\n";
    weights_file.close();

    std::cout << "Results saved to data/" << std::endl;
    std::cout << "\n=== ALL COMPLETED ===" << std::endl;

    return 0;
}
#include "generator/generator.h"
#include "classifier/classifier.h"
#include "geneticalg/geneticalg.h"
#include <iostream>
#include <fstream>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Binary Linear Classifier with GA     " << std::endl;
    std::cout << "  (Бинарный линейный классификатор     " << std::endl;
    std::cout << "   с генетической оптимизацией)        " << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[Этап 1] Генерация обучающей выборки..." << std::endl;

    DataGenerator gen(42);
    auto data = gen.generate(150, 0.5f);  // 150 точек на класс, разделение 0.8
    gen.save_to_csv(data, "data/train.csv");

    std::cout << "  • Сгенерировано точек: " << data.size() << std::endl;
    std::cout << "  • Класс +1: " << data.size() / 2 << " точек" << std::endl;
    std::cout << "  • Класс -1: " << data.size() / 2 << " точек" << std::endl;
    std::cout << "  • Сохранено в: data/train.csv" << std::endl;

    std::cout << "\n[Этап 2] Генетическая оптимизация весов..." << std::endl;

    GeneticAlg ga(42, 50, 100, 0.1f, 0.1f);
    Vector best_weights = ga.train(data);

    std::cout << "\n[Этап 3] Финальная модель классификации..." << std::endl;

    Classifier clf(2);
    clf.set_weights(best_weights);

    std::cout << "\n--- Параметры разделяющей прямой ---" << std::endl;
    std::cout << "  Уравнение: w1*x + w2*y + b = 0" << std::endl;
    std::cout << "  w1 (коэф. при x) = " << best_weights.at(0) << std::endl;
    std::cout << "  w2 (коэф. при y) = " << best_weights.at(1) << std::endl;
    std::cout << "  b  (смещение)    = " << best_weights.at(2) << std::endl;

    float accuracy = clf.accuracy(data);
    std::cout << "\n--- Качество классификации ---" << std::endl;
    std::cout << "  Точность на обучающей выборке: " << accuracy * 100 << "%" << std::endl;

    std::cout << "\n[Этап 4] Экспорт результатов..." << std::endl;

    // Сохраняем предсказания
    std::ofstream results("data/results.csv");
    results << "x,y,true_label,predicted_label\n";
    for (const auto& p : data) {
        results << p.x << "," << p.y << "," << p.label << "," << clf.predict(p) << "\n";
    }
    results.close();

    // Сохраняем веса
    std::ofstream weights_file("data/weights.csv");
    weights_file << "w1,w2,b\n" << best_weights.at(0) << "," << best_weights.at(1) << "," << best_weights.at(2) << "\n";
    weights_file.close();

    std::cout << "  • Предсказания: data/results.csv" << std::endl;
    std::cout << "  • Веса модели: data/weights.csv" << std::endl;

    return 0;
}
#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include "random/random.h"
#include <string>
#include <vector>

// ← ИЗМЕНЕНО: универсальный DataPoint под любое количество признаков
struct DataPoint {
    std::vector<float> features;  // вместо float x, y
    int label;

    // Конструкторы
    DataPoint() : label(0) {}
    DataPoint(std::vector<float> feats, int lbl)
        : features(std::move(feats)), label(lbl) {}

    // Для обратной совместимости: удобные геттеры для 2D
    float x() const { return features.size() > 0 ? features[0] : 0.0f; }
    float y() const { return features.size() > 1 ? features[1] : 0.0f; }
};

class DataGenerator {
private:
    Random rng;

public:
    DataGenerator(unsigned int seed = 42);

    // Генерация 2D данных (для лаб 1-2)
    std::vector<DataPoint> generate(size_t n_per_class, float separation = 2.0f);

    // Сохранение/загрузка
    void save_to_csv(const std::vector<DataPoint>& data, const std::string& filename);
    std::vector<DataPoint> load_from_csv(const std::string& filename);
};

#endif // GENERATOR_HPP
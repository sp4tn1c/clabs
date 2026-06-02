#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "../generator/generator.h"  // ← Для DataPoint
#include <string>
#include <vector>

class DataLoader {
public:
    static std::vector<DataPoint> loadFromCSV(const std::string& filename,
        size_t max_features = 0);
    static void splitTrainTest(const std::vector<DataPoint>& data,
                               std::vector<DataPoint>& train,
                               std::vector<DataPoint>& test,
                               float train_ratio = 0.8f,
                               unsigned seed = 42);
    static size_t getFeatureCount(const std::vector<DataPoint>& data);
};

#endif // DATALOADER_HPP
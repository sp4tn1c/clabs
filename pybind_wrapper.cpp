#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "neural/neuralnetwork.h"
#include "neural/neural_trainer.h"
#include "data/dataloader.h"
#include "metrics/f1score.h"
#include <vector>
#include <iostream>

namespace py = pybind11;

// ==================== Обёртка для NeuralNetwork ====================

class NeuralNetworkWrapper {
private:
    NeuralNetwork* net;
    size_t input_size;
    
public:
    NeuralNetworkWrapper(size_t in_size, size_t hidden_size, size_t out_size) 
        : input_size(in_size) {
        net = new NeuralNetwork(static_cast<int>(in_size), 
                                static_cast<int>(hidden_size), 
                                static_cast<int>(out_size));
        net->init_weights();
    }
    
    ~NeuralNetworkWrapper() {
        delete net;
    }
    
    // Предсказание для одной точки
    float predict(const std::vector<float>& features) {
        return net->predict(features);
    }
    
    // Предсказание для множества точек
    std::vector<int> predict_batch(const std::vector<std::vector<float>>& features_batch) {
        std::vector<int> predictions;
        for (const auto& features : features_batch) {
            float prob = net->predict(features);
            predictions.push_back((prob >= 0.5f) ? 1 : 0);
        }
        return predictions;
    }
    
    // Обучение на данных
    void train(const std::vector<std::vector<float>>& X,
           const std::vector<int>& y,
           int epochs = 100,
           float mutation_rate = 0.2f,
           float mutation_scale = 0.2f) {
        // Конвертируем в DataPoint
        std::vector<DataPoint> data;

        // ← ← ← ОТЛАДКА ← ← ←
        int count_neg1 = 0, count_0 = 0, count_1 = 0;
        for (int l : y) {
            if (l == -1) count_neg1++;
            else if (l == 0) count_0++;
            else if (l == 1) count_1++;
        }
        std::cout << "  [DEBUG] Labels: -1=" << count_neg1
                  << ", 0=" << count_0 << ", 1=" << count_1 << std::endl;
        // ← ← ← КОНЕЦ ОТЛАДКИ ← ← ←

        for (size_t i = 0; i < X.size(); ++i) {
            int label = y[i];
            if (label == -1) label = 0;  // Конвертация
            data.emplace_back(X[i], label);
        }

        NeuralTrainer trainer(*net, data, 50, epochs, mutation_rate, mutation_scale);
        trainer.train();
    }
    
    // Получить веса
    std::vector<float> get_weights() {
        Vector w = net->get_weights();
        std::vector<float> result(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            result[i] = w.at(i);
        }
        return result;
    }
    
    // Установить веса
    void set_weights(const std::vector<float>& weights) {
        Vector w(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            w.at(i) = weights[i];
        }
        net->set_weights(w);
    }
    
    // Получить число входов
    size_t get_input_size() const { return input_size; }
};

// ==================== Функции для работы с данными ====================

std::vector<std::vector<float>> load_csv_features(const std::string& filename) {
    auto data = DataLoader::loadFromCSV(filename);
    std::vector<std::vector<float>> features;
    for (const auto& p : data) {
        features.push_back(p.features);
    }
    return features;
}

std::vector<int> load_csv_labels(const std::string& filename) {
    auto data = DataLoader::loadFromCSV(filename);
    std::vector<int> labels;
    for (const auto& p : data) {
        labels.push_back(p.label);
    }
    return labels;
}

// ==================== Расчёт метрик ====================

py::dict calculate_f1(const std::vector<int>& predicted, 
                      const std::vector<int>& true_labels) {
    auto report = F1Metric::calculate(predicted, true_labels);
    
    py::dict result;
    result["precision"] = report.precision();
    result["recall"] = report.recall();
    result["f1"] = report.f1();
    result["true_positive"] = report.true_positive;
    result["false_positive"] = report.false_positive;
    result["true_negative"] = report.true_negative;
    result["false_negative"] = report.false_negative;
    
    return result;
}

// ==================== PYBIND11 MODULE ====================

PYBIND11_MODULE(ml_model_cpp, m) {
    m.doc() = "ML Labs C++ Module - Neural Network for Classification";
    
    // Класс NeuralNetwork
    py::class_<NeuralNetworkWrapper>(m, "NeuralNetwork")
        .def(py::init<size_t, size_t, size_t>(), 
             py::arg("input_size"), 
             py::arg("hidden_size") = 0,  // 0 = auto (2*input)
             py::arg("output_size") = 1,
             "Create neural network")
        .def("predict", &NeuralNetworkWrapper::predict, 
             "Predict probability for single sample")
        .def("predict_batch", &NeuralNetworkWrapper::predict_batch, 
             "Predict classes for multiple samples")
        .def("train", &NeuralNetworkWrapper::train,
             py::arg("X"), py::arg("y"),
             py::arg("epochs") = 100,
             py::arg("mutation_rate") = 0.2f,
             py::arg("mutation_scale") = 0.2f,
             "Train the network")
        .def("get_weights", &NeuralNetworkWrapper::get_weights,
             "Get network weights")
        .def("set_weights", &NeuralNetworkWrapper::set_weights,
             "Set network weights")
        .def("get_input_size", &NeuralNetworkWrapper::get_input_size,
             "Get input size");
    
    // Функции для работы с данными
    m.def("load_csv_features", &load_csv_features, 
          "Load features from CSV file");
    m.def("load_csv_labels", &load_csv_labels, 
          "Load labels from CSV file");
    
    // Метрики
    m.def("calculate_f1", &calculate_f1, 
          "Calculate F1 score and other metrics");
}
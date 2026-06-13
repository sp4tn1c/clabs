#!/usr/bin/env python3
"""
Lab 3-4: Neural Network Training Script
Uses C++ module via pybind11

Each dataset gets its own network architecture based on feature count.
Full integration: save/load weights, accuracy, architecture info.

Usage: python3 python/train.py
"""

import sys
import os
import csv
import random

# Добавляем путь к cmake-build-debug для импорта ml_model_cpp
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cmake_build_dir = os.path.join(project_root, 'cmake-build-debug')
sys.path.insert(0, cmake_build_dir)

try:
    import ml_model_cpp as ml
except ImportError as e:
    print(f"❌ Error: ml_model_cpp module not found!")
    print(f"   Searched in: {cmake_build_dir}")
    print(f"   {e}")
    sys.exit(1)


def load_csv(filepath):
    """Загрузка данных из CSV"""
    features = []
    labels = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if not row:
                continue
            features.append([float(x) for x in row[:-1]])
            labels.append(int(float(row[-1])))

    return features, labels


def split_train_test(X, y, train_ratio=0.8, seed=42):
    """Разделение на train/test"""
    random.seed(seed)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    split_idx = int(len(X) * train_ratio)

    X_train = list(X_shuffled[:split_idx])
    X_test = list(X_shuffled[split_idx:])
    y_train = list(y_shuffled[:split_idx])
    y_test = list(y_shuffled[split_idx:])

    return X_train, X_test, y_train, y_test


def evaluate_dataset(name, filepath, save_model=False):
    """
    Оценка на одном датасете.
    Создаёт НОВУЮ нейросеть под число признаков этого датасета.

    :param save_model: если True, сохраняет веса модели после обучения
    """
    print(f"\n{'='*50}")
    print(f"  Evaluating: {name}")
    print(f"{'='*50}")

    # Загрузка
    print(f"\n[1] Loading {name}...")
    X, y = load_csv(filepath)
    num_features = len(X[0]) if X else 2
    print(f"  Loaded: {len(X)} samples")
    print(f"  Features: {num_features}")

    # Статистика меток
    count_0 = y.count(0)
    count_1 = y.count(1)
    print(f"  Labels: 0={count_0}, 1={count_1}")

    # Split 80/20
    print(f"\n[2] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.8, 42)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Создание модели (архитектура под этот датасет!)
    print(f"\n[3] Creating Neural Network...")
    input_size = num_features
    hidden_size = input_size * 2

    net = ml.NeuralNetwork(input_size, hidden_size, 1)

    # Показываем архитектуру (новая фича!)
    arch = net.get_architecture()
    print(f"  Architecture: {arch['input']} → {arch['hidden']} → {arch['output']}")
    print(f"  Total weights: {arch['total_weights']}")

    # Обучение
    print(f"\n[4] Training...")
    net.train(X_train, y_train, epochs=200, mutation_rate=0.2, mutation_scale=0.2)

    # Предсказания
    print(f"\n[5] Evaluating...")
    predictions = net.predict_batch(X_test)

    # Метрики через C++ (новая фича!)
    acc = net.accuracy(X_test, y_test)
    metrics = ml.calculate_f1(predictions, y_test)

    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")

    # Сохранение модели (новая фича!)
    if save_model:
        model_path = f"data/{name}_model.weights"
        net.save_weights(model_path)
        print(f"  ✅ Model saved: {model_path}")

        # Тест загрузки (демонстрация)
        net2 = ml.NeuralNetwork(input_size, hidden_size, 1)
        net2.load_weights(model_path)
        acc2 = net2.accuracy(X_test, y_test)
        print(f"  ✅ Model loaded, accuracy verified: {acc2*100:.2f}%")

    return metrics['f1'], len(X_train), len(X_test)


def main():
    print("="*50)
    print("  Lab 3-4: Neural Network (Python + C++)")
    print("  Using pybind11 for C++ integration")
    print("  (Full API: save/load weights, accuracy, architecture)")
    print("="*50)

    results = []

    # d1
    f1_d1, train_d1, test_d1 = evaluate_dataset("d1", "data/d1.csv", save_model=True)
    results.append(("d1", f1_d1, train_d1, test_d1))

    # d2
    f1_d2, train_d2, test_d2 = evaluate_dataset("d2", "data/d2.csv", save_model=True)
    results.append(("d2", f1_d2, train_d2, test_d2))

    # d3 (если есть)
    if os.path.exists("data/d3.csv"):
        f1_d3, train_d3, test_d3 = evaluate_dataset("d3", "data/d3.csv", save_model=True)
        results.append(("d3", f1_d3, train_d3, test_d3))
    else:
        print(f"\n⚠️  d3.csv not found (will be available on defense)")

    # Итоговая таблица
    print(f"\n{'='*50}")
    print("  FINAL RESULTS")
    print(f"{'='*50}")
    print(f"{'Dataset':<10} | {'Train':<6} | {'Test':<6} | {'F1':<8}")
    print("-" * 40)

    for name, f1, train, test in results:
        print(f"{name:<10} | {train:<6} | {test:<6} | {f1:<8.3f}")

    # Итоговая оценка
    if len(results) >= 2:
        final_score = 0.5 * results[0][1] + 0.5 * results[1][1]

        print(f"\n{'='*50}")
        print("  FINAL SCORE (by TЗ formula)")
        print(f"{'='*50}")
        print(f"  Formula: 0.5 × F1(d1) + 0.5 × F1(d2)")
        print(f"  F1(d1) = {results[0][1]:.3f}")
        print(f"  F1(d2) = {results[1][1]:.3f}")
        print(f"\n  >>> FINAL SCORE: {final_score:.3f} <<<")

        if final_score >= 0.55:
            print(f"\n  ✅ PASSED! (>= 0.55)")
        else:
            print(f"\n  ❌ FAILED! (< 0.55)")

    print(f"\n{'='*50}")
    print("  Ready for Defense!")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
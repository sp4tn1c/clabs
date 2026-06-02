#!/usr/bin/env python3
"""
Визуализация результатов Лабораторных 3-4
Запуск: python3 python/plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Проверяем, что файлы существуют
def check_files():
    required = ['data/d1_predictions.csv', 'data/d2_predictions.csv']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"❌ Error: Missing files: {missing}")
        print("   First run the C++ program: ./cmake-build-debug/laba1")
        return False
    return True

def main():
    print("🔍 Checking files...")
    if not check_files():
        return 1

    # Загрузка результатов
    print("\n📊 Loading data...")
    d1 = pd.read_csv('data/d1_predictions.csv')
    d2 = pd.read_csv('data/d2_predictions.csv')

    print(f"Loaded d1: {len(d1)} points, columns: {d1.columns.tolist()}")
    print(f"Loaded d2: {len(d2)} points, columns: {d2.columns.tolist()}")

    # Определяем имена колонок (feature_0, feature_1 или x, y)
    d1_x = 'feature_0' if 'feature_0' in d1.columns else 'x'
    d1_y = 'feature_1' if 'feature_1' in d1.columns else 'y'
    d2_x = 'feature_0' if 'feature_0' in d2.columns else 'x'
    d2_y = 'feature_1' if 'feature_1' in d2.columns else 'y'

    # Создаём фигуру: 2 строки × 2 столбца
    # Строка 1: d1 (2 признака)
    # Строка 2: d2 (4 признака, проекция)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {0: 'blue', 1: 'red', -1: 'blue'}
    labels = {0: 'Class 0', 1: 'Class 1', -1: 'Class 1'}

    # ==================== D1 (2 признака) ====================
    print("\n📈 Plotting d1...")

    # Левое верхнее: реальные классы d1
    ax1 = axes[0, 0]
    for label in [0, 1]:
        subset = d1[d1['true_label'] == label]
        ax1.scatter(subset[d1_x], subset[d1_y],
                    color=colors[label], label=labels[label],
                    alpha=0.7, s=25, edgecolors='black', linewidth=0.5)
    ax1.set_title('Dataset d1: Real classes', fontsize=12)
    ax1.set_xlabel('Feature 0')
    ax1.set_ylabel('Feature 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Правое верхнее: предсказания нейросети d1
    ax2 = axes[0, 1]
    errors = d1[d1['true_label'] != d1['pred_neural']]
    correct = d1[d1['true_label'] == d1['pred_neural']]

    ax2.scatter(correct[d1_x], correct[d1_y],
                c=correct['pred_neural'], cmap='coolwarm',
                vmin=0, vmax=1, alpha=0.7, s=25, label='Correct')
    if not errors.empty:
        ax2.scatter(errors[d1_x], errors[d1_y],
                    c='yellow', edgecolors='black',
                    s=40, marker='x', linewidth=1.5, label='Error')

    ax2.set_title(f'Dataset d1: Neural Network Predictions', fontsize=12)
    ax2.set_xlabel('Feature 0')
    ax2.set_ylabel('Feature 1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ==================== D2 (4 признака, проекция) ====================
    print("📈 Plotting d2...")

    # Левое нижнее: реальные классы d2
    ax3 = axes[1, 0]
    for label in [0, 1]:
        subset = d2[d2['true_label'] == label]
        ax3.scatter(subset[d2_x], subset[d2_y],
                    color=colors[label], label=labels[label],
                    alpha=0.7, s=25, edgecolors='black', linewidth=0.5)
    ax3.set_title('Dataset d2: Real classes (projection)', fontsize=12)
    ax3.set_xlabel('Feature 0')
    ax3.set_ylabel('Feature 1')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Правое нижнее: предсказания нейросети d2
    ax4 = axes[1, 1]
    errors = d2[d2['true_label'] != d2['pred_neural']]
    correct = d2[d2['true_label'] == d2['pred_neural']]

    ax4.scatter(correct[d2_x], correct[d2_y],
                c=correct['pred_neural'], cmap='coolwarm',
                vmin=0, vmax=1, alpha=0.7, s=25, label='Correct')
    if not errors.empty:
        ax4.scatter(errors[d2_x], errors[d2_y],
                    c='yellow', edgecolors='black',
                    s=40, marker='x', linewidth=1.5, label='Error')

    ax4.set_title(f'Dataset d2: Neural Network Predictions (projection)', fontsize=12)
    ax4.set_xlabel('Feature 0')
    ax4.set_ylabel('Feature 1')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()

    # Сохраняем
    os.makedirs('output', exist_ok=True)
    output_path = 'output/lab34_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graph saved to: {output_path}")

    # Показываем график
    try:
        plt.show()
    except:
        print("ℹ️  Display not available, graph saved to file")

    # ==================== СТАТИСТИКА ====================
    print("\n" + "="*50)
    print("📊 STATISTICS")
    print("="*50)

    # D1
    pred_d1 = (d1['pred_neural'] >= 0.5).astype(int)
    acc_d1 = (pred_d1 == d1['true_label']).mean()
    print(f"\nDataset d1:")
    print(f"  Total points: {len(d1)}")
    print(f"  Accuracy: {acc_d1 * 100:.2f}%")
    print(f"  Errors: {(pred_d1 != d1['true_label']).sum()}")

    # D2
    pred_d2 = (d2['pred_neural'] >= 0.5).astype(int)
    acc_d2 = (pred_d2 == d2['true_label']).mean()
    print(f"\nDataset d2:")
    print(f"  Total points: {len(d2)}")
    print(f"  Accuracy: {acc_d2 * 100:.2f}%")
    print(f"  Errors: {(pred_d2 != d2['true_label']).sum()}")

    # Итоговая оценка по формуле из ТЗ
    from sklearn.metrics import f1_score
    f1_d1 = f1_score(d1['true_label'], pred_d1)
    f1_d2 = f1_score(d2['true_label'], pred_d2)
    final_score = 0.5 * f1_d1 + 0.5 * f1_d2

    print("\n" + "="*50)
    print("🎯 FINAL SCORE (by TЗ formula)")
    print("="*50)
    print(f"  F1(d1) = {f1_d1:.3f}")
    print(f"  F1(d2) = {f1_d2:.3f}")
    print(f"  Final: 0.5 × {f1_d1:.3f} + 0.5 × {f1_d2:.3f} = {final_score:.3f}")

    if final_score >= 0.55:
        print(f"\n  ✅ PASSED! (>= 0.55)")
    else:
        print(f"\n  ❌ FAILED! (< 0.55)")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
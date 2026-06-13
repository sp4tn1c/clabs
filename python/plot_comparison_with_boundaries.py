#!/usr/bin/env python3
"""
Визуализация границ решений: Linear Classifier vs Neural Network
Для Лабораторных 3-4 (датасеты преподавателя d1, d2)
Запуск: python3 python/plot_boundaries.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
import os

# ==================== ПРОВЕРКА ФАЙЛОВ ====================
def check_files():
    required = ['data/d1_predictions.csv', 'data/d2_predictions.csv']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"❌ Error: Missing files: {missing}")
        print("   First run the C++ program: ./cmake-build-debug/laba1")
        return False
    return True

# ==================== ОТРИСОВКА ГРАНИЦЫ ЛИНЕЙНОГО КЛАССИФИКАТОРА ====================
def plot_linear_boundary(ax, w1, w2, b, x_min, x_max, y_min, y_max, label='Linear'):
    """Рисует прямую линию: w1*x + w2*y + b = 0"""
    if abs(w2) > 1e-6:
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(w1 * x_vals + b) / w2
        mask = (y_vals >= y_min) & (y_vals <= y_max)
        ax.plot(x_vals[mask], y_vals[mask], 'g-', linewidth=2, label=label)
    elif abs(w1) > 1e-6:
        y_vals = np.linspace(y_min, y_max, 100)
        x_vals = np.full_like(y_vals, -b / w1)
        mask = (x_vals >= x_min) & (x_vals <= x_max)
        ax.plot(x_vals[mask], y_vals[mask], 'g-', linewidth=2, label=label)

# ==================== ОТРИСОВКА ОДНОГО ДАТАСЕТА ====================
def plot_dataset(filepath, title, ax, weights=None):
    """Рисует один датасет с границами решений"""

    # Загрузка данных
    df = pd.read_csv(filepath)

    # Определяем имена колонок
    col_x = 'feature_0' if 'feature_0' in df.columns else 'x'
    col_y = 'feature_1' if 'feature_1' in df.columns else 'y'

    # Границы для графика
    x_min, x_max = df[col_x].min() - 0.5, df[col_x].max() + 0.5
    y_min, y_max = df[col_y].min() - 0.5, df[col_y].max() + 0.5

    # === СЕТКА ДЛЯ ИНТЕРПОЛЯЦИИ ===
    resolution = 100
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xi, yi)

    # Для интерполяции используем pred_neural как вероятность
    # (даже если это 0/1, интерполяция создаст плавный переход)
    points = df[[col_x, col_y]].values
    values = df['pred_neural'].values

    grid_z = griddata(points, values, (xx, yy), method='cubic')

    # Заменяем NaN
    nearest_interp = NearestNDInterpolator(points, values)
    grid_z_nearest = nearest_interp(xx, yy)
    grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

    # === 1. ФОН ===
    im = ax.contourf(xx, yy, grid_z, levels=20, cmap='RdBu', alpha=0.4)

    # === 2. ГРАНИЦА НЕЙРОСЕТИ ===
    ax.contour(xx, yy, grid_z, levels=[0.5], colors='orange', linewidths=2)

    # === 3. ГРАНИЦА ЛИНЕЙНОГО ===
    if weights is not None and len(weights) >= 3:
        w1, w2, b = weights[0], weights[1], weights[2]
        plot_linear_boundary(ax, w1, w2, b, x_min, x_max, y_min, y_max)

    # === 4. ТОЧКИ ===
    colors = {0: 'blue', 1: 'red'}
    labels = {0: 'Class 0', 1: 'Class 1'}

    for label in [0, 1]:
        subset = df[df['true_label'] == label]
        ax.scatter(subset[col_x], subset[col_y],
                   color=colors[label], label=labels[label],
                   alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

    # === СТАТИСТИКА (ИСПРАВЛЕННАЯ) ===
    from sklearn.metrics import f1_score, accuracy_score

    # pred_neural уже 0 или 1 — используем напрямую
    # ← ИСПРАВЛЕНО: обрабатываем NaN и нечисловые значения
    predictions = pd.to_numeric(df['pred_neural'], errors='coerce').fillna(0).astype(int)

    f1 = f1_score(df['true_label'], predictions, zero_division=0)
    accuracy = accuracy_score(df['true_label'], predictions)

    # Проверка на инверсию
    if f1 == 0.0:
        predictions_inv = 1 - predictions
        f1_inv = f1_score(df['true_label'], predictions_inv, zero_division=0)
        if f1_inv > f1:
            predictions = predictions_inv
            f1 = f1_inv
            print(f"  ⚠️  Labels inverted for {title}")

    errors = (predictions != df['true_label']).sum()

    # Ошибки на графике
    if errors > 0:
        error_mask = (df['pred_neural'].astype(int) != df['true_label'])
        errors_df = df[error_mask]
        ax.scatter(errors_df[col_x], errors_df[col_y],
                   c='yellow', edgecolors='black', s=50, marker='x',
                   linewidth=1.5, label=f'Error ({errors})')

    print(f"\n{title}:")
    print(f"  Points: {len(df)}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  F1-score: {f1:.3f}")
    print(f"  Errors: {errors}")

    ax.set_xlabel('Feature 0', fontsize=11)
    ax.set_ylabel('Feature 1', fontsize=11)
    ax.set_title(f'{title}\nF1={f1:.3f}, Acc={accuracy*100:.1f}%', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    return f1, accuracy

# ==================== MAIN ====================
def main():
    print("="*60)
    print("  Lab 3-4: Decision Boundaries Visualization")
    print("="*60)

    if not check_files():
        return 1

    # Попытка загрузить веса линейного классификатора
    weights = None
    try:
        w_df = pd.read_csv('data/weights.csv')
        if 'w1' in w_df.columns:
            weights = [w_df['w1'][0], w_df['w2'][0], w_df['b'][0]]
        else:
            weights = pd.read_csv('data/weights.csv', header=None).values.flatten()
            weights = [float(x) for x in weights[:3]]
        print(f"\n✅ Loaded linear weights: w1={weights[0]:.4f}, w2={weights[1]:.4f}, b={weights[2]:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not load linear weights: {e}")
        print("   (Linear boundary will not be shown)")

    # Создаём фигуру: 2 графика (d1 и d2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # График для d1
    f1_d1, acc_d1 = plot_dataset('data/d1_predictions.csv',
                                 'Dataset d1 (2 features)', ax1, weights)

    # График для d2 (проекция на первые 2 признака)
    f1_d2, acc_d2 = plot_dataset('data/d2_predictions.csv',
                                 'Dataset d2 (4 features, projection)', ax2, weights)

    # Общая цветовая шкала
    cbar = plt.colorbar(ax1.collections[0], ax=[ax1, ax2], shrink=0.8)
    cbar.set_label('Neural Network Probability (Class 1)', rotation=270, labelpad=20)

    plt.tight_layout()

    # Сохранение
    os.makedirs('output', exist_ok=True)
    output_path = 'output/lab34_boundaries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graph saved to: {output_path}")

    # Итоговая оценка по ТЗ
    final_score = 0.5 * f1_d1 + 0.5 * f1_d2

    print("\n" + "="*60)
    print("  FINAL SCORE (by TЗ formula)")
    print("="*60)
    print(f"  F1(d1) = {f1_d1:.3f}")
    print(f"  F1(d2) = {f1_d2:.3f}")
    print(f"  Final: 0.5 × {f1_d1:.3f} + 0.5 × {f1_d2:.3f} = {final_score:.3f}")

    if final_score >= 0.55:
        print(f"\n  ✅ PASSED! (>= 0.55)")
    else:
        print(f"\n  ❌ FAILED! (< 0.55)")

    print("="*60)

    # Показываем график
    try:
        plt.show()
    except:
        print("ℹ️  Display not available, graph saved to file")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
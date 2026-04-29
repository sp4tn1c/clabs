import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Загрузка данных
data = pd.read_csv('data/lab2_results.csv')

# Загрузка весов Lab 1 (пробуем разные варианты)
try:
    weights_lab1 = pd.read_csv('data/weights.csv', header=None).values.flatten()
    # Преобразуем в float, если вдруг строки
    weights_lab1 = [float(x) for x in weights_lab1]
except:
    # Если не получилось, пробуем с заголовком
    weights_lab1 = pd.read_csv('data/weights.csv').values.flatten()
    weights_lab1 = [float(x) for x in weights_lab1]

w1, w2, b = weights_lab1[0], weights_lab1[1], weights_lab1[2]

print(f"Lab 1 weights: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")

# Загрузка весов нейросети (для информации)
try:
    nn_weights = pd.read_csv('data/lab2_weights.csv', header=None).values.flatten()
    nn_weights = [float(x) for x in nn_weights]
    print(f"Neural network weights count: {len(nn_weights)}")
except:
    print("Could not load neural network weights")

# Функция для отрисовки границы решения линейного классификатора
def plot_linear_boundary(ax, w1, w2, b, x_min, x_max, y_min, y_max):
    if abs(w2) > 1e-6:
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(w1 * x_vals + b) / w2
        # Обрезаем, чтобы не выходило за пределы
        mask = (y_vals >= y_min) & (y_vals <= y_max)
        ax.plot(x_vals[mask], y_vals[mask], 'g-', linewidth=2, label='Linear classifier (Lab 1)')
    else:
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = np.full_like(x_vals, -b / w1) if abs(w1) > 1e-6 else np.zeros_like(x_vals)
        mask = (y_vals >= y_min) & (y_vals <= y_max)
        ax.plot(x_vals[mask], y_vals[mask], 'g-', linewidth=2, label='Linear classifier (Lab 1)')

# Определяем границы для графика
x_min, x_max = data['x'].min() - 0.5, data['x'].max() + 0.5
y_min, y_max = data['y'].min() - 0.5, data['y'].max() + 0.5

# Создаём сетку для интерполяции
resolution = 100
xi = np.linspace(x_min, x_max, resolution)
yi = np.linspace(y_min, y_max, resolution)
xx, yy = np.meshgrid(xi, yi)

# Интерполяция предсказаний нейросети
points = data[['x', 'y']].values
values = data['prediction'].values

print("Интерполяция...")
grid_z = griddata(points, values, (xx, yy), method='cubic')

# Заменяем NaN на ближайшее значение (для краёв)
from scipy.interpolate import NearestNDInterpolator
nearest_interp = NearestNDInterpolator(points, values)
grid_z_nearest = nearest_interp(xx, yy)
grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

# Создаём график
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# 1. Рисуем фон с вероятностями нейросети
im = ax.contourf(xx, yy, grid_z, levels=20, cmap='RdBu', alpha=0.4)

# 2. Рисуем границу нейросети (где prediction = 0.5)
contour = ax.contour(xx, yy, grid_z, levels=[0.5], colors='orange', linewidths=2, label='Neural network boundary')

# 3. Рисуем линейную границу
plot_linear_boundary(ax, w1, w2, b, x_min, x_max, y_min, y_max)

# 4. Рисуем точки
colors = {0: 'blue', 1: 'red'}
labels = {0: 'Class 0', 1: 'Class 1'}

for label in [0, 1]:
    subset = data[data['label'] == label]
    ax.scatter(subset['x'], subset['y'], 
               color=colors[label], label=labels[label], 
               alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Comparison: Linear Classifier vs Neural Network', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')

# Добавляем цветовую шкалу для вероятностей
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Neural network probability of class 1', rotation=270, labelpad=20)

# Добавляем легенду для контура (если нужно)
handles, labels = ax.get_legend_handles_labels()
from matplotlib.lines import Line2D
handles.append(Line2D([0], [0], color='orange', linewidth=2, label='Neural network boundary'))
ax.legend(handles=handles, loc='best')

plt.tight_layout()
plt.savefig('output/lab1_vs_lab2_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nГрафик сохранён в output/lab1_vs_lab2_boundaries.png")

# Дополнительная статистика
predictions = (data['prediction'] >= 0.5).astype(int)
accuracy = (predictions == data['label']).mean()
print(f"Точность нейросети: {accuracy * 100:.2f}%")
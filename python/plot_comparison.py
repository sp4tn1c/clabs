import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка результатов Lab 2
lab2 = pd.read_csv('data/lab2_results.csv')

print(f"Загружено точек: {len(lab2)}")
print(f"Колонки: {lab2.columns.tolist()}")

# Создаём график
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Левое поле: реальные классы
ax1 = axes[0]
colors = {0: 'blue', 1: 'red'}
labels = {0: 'Class 0', 1: 'Class 1'}

for label in [0, 1]:
    subset = lab2[lab2['label'] == label]
    ax1.scatter(subset['x'], subset['y'], 
                color=colors[label], label=labels[label], 
                alpha=0.7, s=25)
ax1.set_title('Real classes', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Правое поле: предсказания нейросети
ax2 = axes[1]
scatter = ax2.scatter(lab2['x'], lab2['y'], 
                      c=lab2['prediction'], cmap='coolwarm', 
                      vmin=0, vmax=1, alpha=0.7, s=25)
ax2.set_title('Neural network predictions (probability)', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Probability of class 1')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()

# Сохраняем
plt.savefig('output/lab2_comparison.png', dpi=150, bbox_inches='tight')
print("\nГрафик сохранён в output/lab2_comparison.png")

# Показываем график
plt.show()

# Дополнительная статистика
predictions = (lab2['prediction'] >= 0.5).astype(int)
accuracy = (predictions == lab2['label']).mean()
print(f"\nТочность нейросети: {accuracy * 100:.2f}%")
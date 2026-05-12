# src/train.py
"""
Обучение модели на реальных данных Goodreads Books (11 117 книг)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

print("=" * 60)
print("🚀 ПРОГНОЗИРОВАНИЕ РЕЙТИНГА КНИГ GOODREADS")
print("=" * 60)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
print("\n1. ЗАГРУЗКА ДАННЫХ")
print("-" * 40)

df = pd.read_csv('../data/books.csv')
print(f"✅ Загружено {len(df)} книг")

# Переименовываем колонку с лишними пробелами
df.rename(columns={'  num_pages': 'num_pages'}, inplace=True)

# ============================================
# 2. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================
print("\n2. ПОДГОТОВКА ПРИЗНАКОВ")
print("-" * 40)

# Признаки
feature_columns = ['num_pages', 'ratings_count', 'text_reviews_count']
target_column = 'average_rating'

# Извлекаем год из publication_date
df['publication_year'] = pd.to_datetime(df['publication_date'], errors='coerce').dt.year
feature_columns.append('publication_year')

# Удаляем пропуски
df_clean = df[feature_columns + [target_column]].dropna()
print(f"✅ После очистки: {len(df_clean)} книг")

X = df_clean[feature_columns]
y = df_clean[target_column]

print(f"\n🔹 Признаки (X): {', '.join(feature_columns)}")
print(f"🔹 Цель (y): {target_column}")

# ============================================
# 3. МАСШТАБИРОВАНИЕ И РАЗДЕЛЕНИЕ
# ============================================
print("\n3. МАСШТАБИРОВАНИЕ И РАЗДЕЛЕНИЕ")
print("-" * 40)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"✅ Обучение: {len(X_train)} книг (80%)")
print(f"✅ Тест (сейф): {len(X_test)} книг (20%)")

# ============================================
# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================
print("\n4. ОБУЧЕНИЕ МОДЕЛЕЙ")
print("-" * 40)

# Линейная регрессия
print("\n📈 LinearRegression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"   MAE: {lr_mae:.4f}")
print(f"   R²:  {lr_r2:.4f}")

# Случайный лес
print("\n🌲 RandomForestRegressor...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"   MAE: {rf_mae:.4f}")
print(f"   R²:  {rf_r2:.4f}")

# ============================================
# 5. КРОСС-ВАЛИДАЦИЯ
# ============================================
print("\n5. КРОСС-ВАЛИДАЦИЯ (5-fold)")
print("-" * 40)

lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
rf_cv = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')

print(f"LinearRegression: R² = {lr_cv.mean():.4f} (±{lr_cv.std():.4f})")
print(f"RandomForest:     R² = {rf_cv.mean():.4f} (±{rf_cv.std():.4f})")

# ============================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================
print("\n6. ВИЗУАЛИЗАЦИЯ ОШИБОК")
print("-" * 40)

errors = y_test - lr_pred

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.3, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Фактический рейтинг")
plt.ylabel("Предсказанный рейтинг")
plt.title(f"LinearRegression: предсказания vs факт\nMAE = {lr_mae:.3f}")

plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel("Ошибка предсказания")
plt.ylabel("Количество книг")
plt.title(f"Распределение ошибок\nсредняя = {errors.mean():.3f}, std = {errors.std():.3f}")

plt.tight_layout()
os.makedirs('../models', exist_ok=True)
plt.savefig('../models/error_plot.png', dpi=150)
print("✅ График сохранен: models/error_plot.png")

# ============================================
# 7. ВЫБОР И СОХРАНЕНИЕ МОДЕЛИ
# ============================================
print("\n7. ВЫБОР И СОХРАНЕНИЕ МОДЕЛИ")
print("-" * 40)

if lr_mae <= rf_mae:
    best_model = lr
    best_name = "LinearRegression"
    best_mae = lr_mae
    best_r2 = lr_r2
else:
    best_model = rf
    best_name = "RandomForestRegressor"
    best_mae = rf_mae
    best_r2 = rf_r2

print(f"✅ Лучшая модель: {best_name}")
print(f"✅ MAE: {best_mae:.4f}")
print(f"✅ R²:  {best_r2:.4f}")

# Сохраняем
joblib.dump(best_model, '../models/best_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
print("✅ Модель сохранена: models/best_model.pkl")
print("✅ Scaler сохранен: models/scaler.pkl")

# ============================================
# 8. ИТОГОВЫЙ ОТЧЕТ
# ============================================
print("\n" + "=" * 60)
print("📊 ИТОГОВЫЙ ОТЧЕТ")
print("=" * 60)

print(f"""
РЕЗУЛЬТАТЫ НА {len(df_clean)} КНИГАХ:

┌─────────────────────────────────────────────────────────────┐
│  Модель: {best_name:<30} │
│  MAE: {best_mae:.4f} звезды (ошибка в {best_mae:.2f} балла)     │
│  R²:  {best_r2:.4f}                                         │
└─────────────────────────────────────────────────────────────┘

ИНТЕРПРЕТАЦИЯ:
• Модель ошибается в среднем на {best_mae:.2f} звезды
• Это {'отличный' if best_mae < 0.25 else 'хороший'} результат

СОЗДАННЫЕ ФАЙЛЫ:
• models/best_model.pkl - обученная модель
• models/scaler.pkl - scaler
• models/error_plot.png - визуализация ошибок
""")

print("\n✅ Готово!")
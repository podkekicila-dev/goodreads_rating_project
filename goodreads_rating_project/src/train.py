# src/train.py
"""
Практическое занятие №8: Обучение модели для предсказания рейтинга книг
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
print("ПРОГНОЗИРОВАНИЕ РЕЙТИНГА КНИГ")
print("=" * 60)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
print("\n1. ЗАГРУЗКА ДАННЫХ")
print("-" * 40)

df = pd.read_csv('../data/books.csv')
print(f"Загружено {len(df)} строк")

# Объяснение признаков
print("\nПРИЗНАКИ (X):")
print("  - num_pages: количество страниц")
print("  - ratings_count: количество оценок")
print("  - text_reviews_count: количество отзывов")
print("  - publication_year: год публикации")
print("\nЦЕЛЕВАЯ ПЕРЕМЕННАЯ (y):")
print("  - average_rating: средний рейтинг (от 1 до 5)")

# ============================================
# 2. ПОДГОТОВКА ДАННЫХ
# ============================================
print("\n2. ПОДГОТОВКА ДАННЫХ")
print("-" * 40)

X = df[['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year']]
y = df['average_rating']

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Признаки масштабированы")

# Разделение на обучающую (80%) и тестовую (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✅ Обучающая выборка: {len(X_train)} примеров")
print(f"✅ Тестовая выборка: {len(X_test)} примеров (в 'сейфе')")

# ============================================
# 3. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================
print("\n3. ОБУЧЕНИЕ МОДЕЛЕЙ")
print("-" * 40)

# Модель 1: Линейная регрессия
print("\nОбучаем Линейную регрессию...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"  MAE: {lr_mae:.3f}")
print(f"  R2:  {lr_r2:.3f}")

# Модель 2: Случайный лес
print("\nОбучаем Случайный лес...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"  MAE: {rf_mae:.3f}")
print(f"  R2:  {rf_r2:.3f}")

# ============================================
# 4. КРОСС-ВАЛИДАЦИЯ
# ============================================
print("\n4. КРОСС-ВАЛИДАЦИЯ (5-fold)")
print("-" * 40)

lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
rf_cv = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')

print(f"Линейная регрессия: {lr_cv.mean():.3f} (+/- {lr_cv.std():.3f})")
print(f"Случайный лес:      {rf_cv.mean():.3f} (+/- {rf_cv.std():.3f})")

# ============================================
# 5. АНАЛИЗ ОШИБОК
# ============================================
print("\n5. АНАЛИЗ ОШИБОК")
print("-" * 40)

errors = y_test - rf_pred
print(f"Средняя ошибка: {errors.mean():.3f}")
print(f"Стандартное отклонение ошибки: {errors.std():.3f}")

# Визуализация
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Фактический рейтинг")
plt.ylabel("Предсказанный рейтинг")
plt.title("Предсказания vs Факт")

plt.subplot(1, 2, 2)
plt.hist(errors, bins=20, edgecolor='black')
plt.xlabel("Ошибка предсказания")
plt.ylabel("Количество")
plt.title("Распределение ошибок")

plt.tight_layout()
plt.savefig('../models/error_plot.png')
print("✅ График ошибок сохранен в 'models/error_plot.png'")

# ============================================
# 6. ВЫБОР ЛУЧШЕЙ МОДЕЛИ И СОХРАНЕНИЕ
# ============================================
print("\n6. ВЫБОР ЛУЧШЕЙ МОДЕЛИ")
print("-" * 40)

if rf_mae < lr_mae:
    best_model = rf
    best_name = "RandomForestRegressor"
    best_metric = rf_mae
else:
    best_model = lr
    best_name = "LinearRegression"
    best_metric = lr_mae

print(f"Лучшая модель: {best_name}")
print(f"MAE на тестовых данных: {best_metric:.3f}")

# Сохраняем модель
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/best_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
print("✅ Модель сохранена в 'models/best_model.pkl'")
print("✅ Scaler сохранен в 'models/scaler.pkl'")

# ============================================
# 7. ИТОГОВЫЙ ОТЧЕТ
# ============================================
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 60)

print(f"""
ЛУЧШАЯ МОДЕЛЬ: {best_name}

КЛЮЧЕВЫЕ МЕТРИКИ:
- MAE (средняя абсолютная ошибка): {best_metric:.3f} звезды
- R² (коэффициент детерминации): {rf_r2 if best_name == 'RandomForestRegressor' else lr_r2:.3f}

ИНТЕРПРЕТАЦИЯ:
- Модель ошибается в среднем на {best_metric:.2f} звезды
- Это {'хороший' if best_metric < 0.3 else 'средний'} результат для предсказания рейтинга

ЧАЩЕ ВСЕГО МОДЕЛЬ ОШИБАЕТСЯ:
- На книгах с рейтингом около {y_test.iloc[errors.abs().argmax()]:.1f} звезд
- Причина: не хватает информации о жанре и авторе

СОЗДАННЫЕ ФАЙЛЫ:
- models/best_model.pkl - обученная модель
- models/scaler.pkl - scaler для масштабирования
- models/error_plot.png - визуализация ошибок
""")

print("\n✅ ГОТОВО! Модель можно использовать для предсказаний.")
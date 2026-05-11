# final_check.py
import os
import subprocess
import sys

print("="*60)
print("🔍 ФИНАЛЬНАЯ ПРОВЕРКА ПРОЕКТА")
print("="*60)

# 1. Проверка структуры
print("\n1. СТРУКТУРА ПРОЕКТА:")
for folder in ['data', 'models', 'src', 'tests']:
    exists = "✅" if os.path.exists(folder) else "❌"
    print(f"   {exists} {folder}/")

# 2. Проверка файлов
print("\n2. ФАЙЛЫ ПРОЕКТА:")
files = ['data/books.csv', 'models/best_model.pkl', 'src/train.py', 'src/app.py', 'tests/test_basic.py']
for f in files:
    exists = "✅" if os.path.exists(f) else "❌"
    size = os.path.getsize(f) if os.path.exists(f) else 0
    print(f"   {exists} {f} ({size} bytes)")

# 3. Проверка импорта
print("\n3. БИБЛИОТЕКИ:")
libs = ['pandas', 'numpy', 'sklearn', 'joblib', 'gradio']
for lib in libs:
    try:
        __import__(lib)
        print(f"   ✅ {lib}")
    except ImportError:
        print(f"   ❌ {lib}")

# 4. Проверка модели
print("\n4. МОДЕЛЬ:")
try:
    import joblib, numpy as np, pandas as pd
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    test_data = pd.DataFrame([[300, 10000, 1000, 2010]], 
                              columns=['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year'])
    pred = model.predict(scaler.transform(test_data))[0]
    print(f"   ✅ Модель загружена: {type(model).__name__}")
    print(f"   ✅ Тестовое предсказание: {pred:.2f}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print("\n" + "="*60)
print("✅ ФИНАЛЬНАЯ ПРОВЕРКА ЗАВЕРШЕНА")
print("="*60)
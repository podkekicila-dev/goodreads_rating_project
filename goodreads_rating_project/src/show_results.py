# src/show_results.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загружаем данные
df = pd.read_csv('../data/books.csv')
X = df[['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year']]
y = df['average_rating']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Загружаем модель
model = joblib.load('../models/best_model.pkl')
predictions = model.predict(X_test)
errors = y_test - predictions

print("=" * 60)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 60)
print(f"""
ЛУЧШАЯ МОДЕЛЬ: LinearRegression

КЛЮЧЕВЫЕ МЕТРИКИ:
- MAE: {abs(errors).mean():.3f} звезды
- RMSE: {np.sqrt((errors**2).mean()):.3f}
- R²: {1 - (errors**2).sum() / ((y_test - y_test.mean())**2).sum():.3f}

СРЕДНЯЯ ОШИБКА: {errors.mean():.3f}

СОЗДАННЫЕ ФАЙЛЫ:
- models/best_model.pkl
- models/scaler.pkl  
- models/error_plot.png
""")
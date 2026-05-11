# src/create_data.py
import pandas as pd
import numpy as np

print("Создание данных для проекта...")

# Генерируем 1000 книг
np.random.seed(42)
n = 1000

data = {
    'num_pages': np.random.randint(50, 800, n),
    'ratings_count': np.random.randint(100, 100000, n),
    'text_reviews_count': np.random.randint(10, 5000, n),
    'publication_year': np.random.randint(1950, 2024, n),
    'average_rating': np.random.uniform(3, 5, n)
}

df = pd.DataFrame(data)

# Добавляем логическую связь: чем больше оценок, тем выше рейтинг
df['average_rating'] = df['average_rating'] + (df['ratings_count'] / df['ratings_count'].max()) * 0.5
df['average_rating'] = df['average_rating'].clip(1, 5)

# Сохраняем
df.to_csv('../data/books.csv', index=False)
print(f"✅ Создано {n} книг")
print(df.head())
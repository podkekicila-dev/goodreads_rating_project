import kagglehub
import pandas as pd
import os
import shutil

print("Скачивание датасета Goodreads Books...")

# Скачиваем датасет
path = kagglehub.dataset_download("jealousleopard/goodreadsbooks")

# Путь к файлу
source_file = os.path.join(path, "books.csv")

# Создаем папку data
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Копируем файл
destination = os.path.join(data_dir, 'goodreads_books.csv')
shutil.copy2(source_file, destination)

# Проверяем
df = pd.read_csv(destination)
print(f"\n✅ Успешно! Датасет загружен.")
print(f"   Строк: {df.shape[0]}")
print(f"   Колонок: {df.shape[1]}")
print(f"   Первые 5 строк:")
print(df.head())
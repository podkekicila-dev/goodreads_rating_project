# src/download_real_data.py
import kagglehub
import pandas as pd
import os
import shutil
import csv

print("=" * 60)
print("📥 СКАЧИВАНИЕ РЕАЛЬНОГО ДАТАСЕТА GOODREADS BOOKS")
print("=" * 60)

# Скачиваем датасет
print("\n1. Скачивание датасета с Kaggle...")
try:
    path = kagglehub.dataset_download("jealousleopard/goodreadsbooks")
    print(f"   ✅ Датасет скачан в: {path}")
except Exception as e:
    print(f"   ❌ Ошибка скачивания: {e}")
    exit(1)

# Путь к исходному файлу
source_file = os.path.join(path, "books.csv")
print(f"\n2. Поиск файла: {source_file}")

if not os.path.exists(source_file):
    print(f"   ❌ Файл books.csv не найден!")
    exit(1)

# Создаем папку data в корне проекта
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

# Копируем оригинал
original_dest = os.path.join(data_dir, 'books_original.csv')
shutil.copy2(source_file, original_dest)
print(f"\n3. Оригинал сохранен: {original_dest}")

# Пробуем прочитать CSV с разными параметрами
print("\n4. Обработка CSV файла...")

# Сначала определим реальный разделитель
with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
    first_line = f.readline()
    print(f"   Первая строка: {first_line[:100]}...")

# Пробуем разные способы чтения
ways_to_read = [
    {'engine': 'python', 'on_bad_lines': 'skip'},
    {'engine': 'python', 'on_bad_lines': 'warn'},
    {'encoding': 'latin1', 'engine': 'python', 'on_bad_lines': 'skip'},
    {'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip', 'quotechar': '"'},
]

df = None
for i, params in enumerate(ways_to_read):
    print(f"   Попытка {i+1}: {params}")
    try:
        df = pd.read_csv(source_file, **params)
        print(f"   ✅ Успешно! Загружено {len(df)} строк, {len(df.columns)} колонок")
        break
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

if df is None:
    print("\n⚠️ Не удалось прочитать CSV стандартными методами.")
    print("   Создаем чистую версию вручную...")
    
    # Ручная очистка - читаем построчно
    clean_rows = []
    with open(source_file, 'r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        print(f"   Заголовки: {headers[:8]}")
        
        for row_num, row in enumerate(reader):
            if len(row) == len(headers):
                clean_rows.append(row)
            if row_num % 1000 == 0:
                print(f"   Обработано {row_num} строк...")
    
    df = pd.DataFrame(clean_rows, columns=headers)
    print(f"   ✅ Создано {len(df)} чистых строк")

# Сохраняем чистую версию
clean_dest = os.path.join(data_dir, 'books.csv')
df.to_csv(clean_dest, index=False)
print(f"\n5. Чистая версия сохранена: {clean_dest}")

# Анализ данных
print("\n" + "=" * 60)
print("📊 АНАЛИЗ ДАТАСЕТА")
print("=" * 60)

print(f"\nКоличество книг: {len(df)}")
print(f"Количество колонок: {len(df.columns)}")
print(f"\nКолонки: {df.columns.tolist()}")

# Проверяем наличие целевой переменной
if 'average_rating' in df.columns:
    print(f"\n✅ Целевая переменная 'average_rating' найдена!")
    print(f"   Диапазон рейтингов: {df['average_rating'].min():.2f} - {df['average_rating'].max():.2f}")
    print(f"   Средний рейтинг: {df['average_rating'].mean():.2f}")
else:
    print(f"\n⚠️ Колонка 'average_rating' не найдена")
    print(f"   Доступные колонки: {df.columns.tolist()[:10]}")

# Информация о пропусках
print(f"\nПропуски в данных:")
for col in df.columns[:8]:
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"   {col}: {missing} ({missing/len(df)*100:.1f}%)")

# Выборочные данные
print(f"\nПервые 3 строки:")
print(df.head(3).to_string())

print("\n" + "=" * 60)
print("✅ ГОТОВО! Датасет загружен и обработан.")
print("=" * 60)
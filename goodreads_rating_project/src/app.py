# src/app.py
"""
Веб-интерфейс для предсказания рейтинга книг
Практическое занятие №9: Прототипирование интерфейса
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# Загрузка модели и scaler
print("Загрузка модели...")

# Определяем пути
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'best_model.pkl')
scaler_path = os.path.join(current_dir, '..', 'models', 'scaler.pkl')

# Загружаем модель
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("✅ Модель загружена!")

# Функция предсказания
def predict_rating(num_pages, ratings_count, text_reviews_count, publication_year):
    """
    Предсказание рейтинга книги на основе характеристик
    
    Args:
        num_pages: количество страниц
        ratings_count: количество оценок
        text_reviews_count: количество текстовых отзывов
        publication_year: год публикации
    
    Returns:
        tuple: (предсказанный рейтинг, описание)
    """
    
    # Создаем DataFrame с входными данными
    input_data = pd.DataFrame({
        'num_pages': [num_pages],
        'ratings_count': [ratings_count],
        'text_reviews_count': [text_reviews_count],
        'publication_year': [publication_year]
    })
    
    # Масштабируем признаки
    input_scaled = scaler.transform(input_data)
    
    # Делаем предсказание
    prediction = model.predict(input_scaled)[0]
    
    # Ограничиваем рейтинг диапазоном 1-5
    prediction = np.clip(prediction, 1, 5)
    
    # Создаем описание
    description = f"""
    📊 **Анализ книги:**
    - 📖 Страниц: {num_pages}
    - ⭐ Оценок: {ratings_count:,}
    - 💬 Отзывов: {text_reviews_count:,}
    - 📅 Год: {publication_year}
    
    🎯 **Результат:**
    - Предсказанный рейтинг: **{prediction:.2f}** / 5.0
    
    📈 **Интерпретация:**
    """
    
    if prediction >= 4.5:
        description += " Отличная книга! Скорее всего, бестселлер."
    elif prediction >= 4.0:
        description += " Хорошая книга, рекомендуется к прочтению."
    elif prediction >= 3.5:
        description += " Неплохая книга, средний уровень."
    elif prediction >= 3.0:
        description += " Посредственная книга, есть над чем работать."
    else:
        description += " Слабый рейтинг, возможно, стоит пропустить."
    
    return round(prediction, 2), description


# Функция для демонстрации с примерами
def predict_with_example(example_num):
    """
    Предсказание для выбранного примера
    """
    examples = {
        1: (320, 50000, 2000, 2020),
        2: (150, 1000, 100, 2015),
        3: (800, 100000, 10000, 2005),
        4: (250, 500, 20, 2023),
        5: (600, 25000, 1500, 2010)
    }
    
    num_pages, ratings_count, text_reviews_count, publication_year = examples[example_num]
    return predict_rating(num_pages, ratings_count, text_reviews_count, publication_year)


# Создание интерфейса Gradio
with gr.Blocks(title="Book Rating Predictor", theme=gr.themes.Soft()) as demo:
    
    # Заголовок
    gr.Markdown("""
    # 📚 Предсказание рейтинга книг Goodreads
    
    ### Введите характеристики книги, и модель предскажет её ожидаемый рейтинг
    Модель обучена на данных о 1000 книгах и предсказывает рейтинг от 1 до 5 звезд.
    """)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Входные параметры
            num_pages = gr.Slider(
                label="📖 Количество страниц",
                minimum=1,
                maximum=1500,
                step=10,
                value=300,
                info="Сколько страниц в книге?"
            )
            
            ratings_count = gr.Number(
                label="⭐ Количество оценок",
                value=10000,
                precision=0,
                info="Сколько читателей оценили книгу?"
            )
            
            text_reviews_count = gr.Number(
                label="💬 Количество текстовых отзывов",
                value=1000,
                precision=0,
                info="Сколько оставили письменных отзывов?"
            )
            
            publication_year = gr.Slider(
                label="📅 Год публикации",
                minimum=1950,
                maximum=2024,
                step=1,
                value=2010,
                info="Когда книга была издана?"
            )
            
            # Кнопка предсказания
            predict_btn = gr.Button("🔮 Предсказать рейтинг", variant="primary")
        
        with gr.Column(scale=1):
            # Результаты
            gr.Markdown("### 📊 Результат предсказания")
            rating_output = gr.Number(
                label="Предсказанный рейтинг",
                precision=2,
                interactive=False
            )
            description_output = gr.Markdown("### 📝 Анализ\n_Заполните параметры и нажмите \"Предсказать\"_")
    
    gr.Markdown("---")
    
    # Примеры книг
    gr.Markdown("### 📖 Попробуйте примеры книг:")
    
    with gr.Row():
        example1 = gr.Button("📘 Пример 1: Современный бестселлер")
        example2 = gr.Button("📗 Пример 2: Малоизвестная книга")
        example3 = gr.Button("📕 Пример 3: Толстый классический роман")
        example4 = gr.Button("📙 Пример 4: Новинка")
        example5 = gr.Button("📔 Пример 5: Среднестатистическая книга")
    
    # Привязка функций
    predict_btn.click(
        fn=predict_rating,
        inputs=[num_pages, ratings_count, text_reviews_count, publication_year],
        outputs=[rating_output, description_output]
    )
    
    # Примеры
    example1.click(lambda: predict_with_example(1), outputs=[rating_output, description_output])
    example2.click(lambda: predict_with_example(2), outputs=[rating_output, description_output])
    example3.click(lambda: predict_with_example(3), outputs=[rating_output, description_output])
    example4.click(lambda: predict_with_example(4), outputs=[rating_output, description_output])
    example5.click(lambda: predict_with_example(5), outputs=[rating_output, description_output])
    
    # Автоматическое обновление при изменении параметров (опционально)
    gr.Markdown("---")
    gr.Markdown("""
    ### ℹ️ Как это работает?
    
    1. Модель обучена на **1000 книгах** из Goodreads
    2. Использует **линейную регрессию** для предсказания
    3. Учитывает: количество страниц, оценок, отзывов и год публикации
    4. Точность модели: **MAE ≈ 0.47 звезды**
    """)

# Запуск приложения
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 ЗАПУСК ВЕБ-ИНТЕРФЕЙСА")
    print("="*50)
    print("\nОткрывается интерфейс для предсказания рейтинга книг...")
    print("После запуска перейдите по ссылке: http://127.0.0.1:7860")
    print("Для остановки нажмите Ctrl+C в терминале\n")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
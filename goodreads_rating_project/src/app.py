# src/app.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

print("📚 Загрузка модели предсказания рейтинга книг...")

# Загружаем модель
model = joblib.load('../models/best_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

print("✅ Модель загружена!")

def predict_rating(num_pages, ratings_count, text_reviews_count, publication_year):
    """Предсказание рейтинга книги"""
    
    input_data = pd.DataFrame([[num_pages, ratings_count, text_reviews_count, publication_year]], 
                              columns=['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year'])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction = np.clip(prediction, 1, 5)
    
    if prediction >= 4.5:
        verdict = "🔮 Отличная книга! Скорее всего, бестселлер."
    elif prediction >= 4.0:
        verdict = "📖 Хорошая книга, рекомендуется к прочтению."
    elif prediction >= 3.5:
        verdict = "📚 Неплохая книга, средний уровень."
    elif prediction >= 3.0:
        verdict = "📘 Посредственная книга, есть над чем работать."
    else:
        verdict = "⚠️ Слабый рейтинг, возможно, стоит пропустить."
    
    return round(prediction, 2), verdict

# Интерфейс
with gr.Blocks(title="Goodreads Book Rating Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 Goodreads Book Rating Predictor
    
    ### Предсказание рейтинга книги на основе её характеристик
    Модель обучена на **11 117 реальных книгах** из Goodreads.
    """)
    
    with gr.Row():
        with gr.Column():
            num_pages = gr.Slider(1, 1500, value=300, label="📖 Количество страниц")
            ratings_count = gr.Number(value=10000, label="⭐ Количество оценок")
            text_reviews_count = gr.Number(value=1000, label="💬 Количество отзывов")
            publication_year = gr.Slider(1950, 2024, value=2010, label="📅 Год публикации")
            predict_btn = gr.Button("🔮 Предсказать рейтинг", variant="primary")
        
        with gr.Column():
            rating_output = gr.Number(label="Предсказанный рейтинг", precision=2)
            verdict_output = gr.Markdown("### 📝 Результат")
    
    predict_btn.click(
        predict_rating,
        inputs=[num_pages, ratings_count, text_reviews_count, publication_year],
        outputs=[rating_output, verdict_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    ### ℹ️ О модели
    - **Алгоритм:** LinearRegression
    - **Точность (MAE):** 0.22 звезды
    - **Обучающая выборка:** 8 893 книги
    - **Признаки:** страницы, оценки, отзывы, год публикации
    """)

if __name__ == "__main__":
    demo.launch()
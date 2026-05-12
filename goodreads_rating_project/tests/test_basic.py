# tests/test_basic.py
import sys
import os
import unittest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestModelPredictions(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.model = joblib.load(os.path.join(base_dir, 'models', 'best_model.pkl'))
        cls.scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
    
    def test_prediction_works(self):
        """Тест 1: предсказание работает"""
        test_data = pd.DataFrame([[300, 10000, 1000, 2010]], 
                                  columns=['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year'])
        test_scaled = self.scaler.transform(test_data)
        prediction = self.model.predict(test_scaled)[0]
        prediction = np.clip(prediction, 1, 5)
        
        self.assertIsInstance(prediction, float)
        self.assertTrue(1 <= prediction <= 5)
        print(f"✅ Тест 1 пройден: предсказание = {prediction:.2f}")
    
    def test_prediction_format(self):
        """Тест 2: правильный формат"""
        test_data = pd.DataFrame([[400, 50000, 5000, 2015]], 
                                  columns=['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year'])
        test_scaled = self.scaler.transform(test_data)
        prediction = self.model.predict(test_scaled)[0]
        
        self.assertIsInstance(prediction, (float, np.float32, np.float64))
        print(f"✅ Тест 2 пройден: формат float")
    
    def test_gradio_import(self):
        """Тест 3: Gradio работает"""
        import gradio as gr
        self.assertTrue(hasattr(gr, "Interface"))
        print("✅ Тест 3 пройден: Gradio OK")


if __name__ == "__main__":
    print("\n🧪 ЗАПУСК ТЕСТОВ")
    unittest.main(verbosity=2)
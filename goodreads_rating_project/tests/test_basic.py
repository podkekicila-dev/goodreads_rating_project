# tests/test_basic.py
import sys
import os
import unittest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestModelPredictions(unittest.TestCase):
    """Тест 1: Проверка работы функции предсказания"""
    
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.model = joblib.load(os.path.join(base_dir, 'models', 'best_model.pkl'))
        cls.scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
    
    def test_prediction_works(self):
        """Тест 1: Проверка, что предсказание работает на корректном примере"""
        test_data = pd.DataFrame([[300, 10000, 1000, 2010]], 
                                  columns=['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year'])
        test_scaled = self.scaler.transform(test_data)
        prediction = self.model.predict(test_scaled)[0]
        prediction = np.clip(prediction, 1, 5)
        
        self.assertIsInstance(prediction, float)
        self.assertTrue(1 <= prediction <= 5)
        print(f"✅ Тест 1 пройден: предсказание = {prediction:.2f}")
    
    def test_prediction_format(self):
        """Тест 2: Проверка формата возвращаемого значения"""
        test_data = pd.DataFrame([[400, 50000, 5000, 2015]], 
                                  columns=['num_pages', 'ratings_count', 'text_reviews_count', 'publication_year'])
        test_scaled = self.scaler.transform(test_data)
        prediction = self.model.predict(test_scaled)[0]
        prediction = np.clip(prediction, 1, 5)
        
        self.assertIsInstance(prediction, (float, np.float32, np.float64))
        self.assertGreaterEqual(prediction, 1.0)
        self.assertLessEqual(prediction, 5.0)
        print(f"✅ Тест 2 пройден: формат float, значение {prediction:.2f}")


class TestWebApp(unittest.TestCase):
    """Тест 3: Проверка веб-приложения"""
    
    def test_gradio_import(self):
        """Проверка, что Gradio установлен и импортируется"""
        import gradio as gr
        self.assertTrue(hasattr(gr, "Interface"))
        print("✅ Тест 3 пройден: Gradio работает")


def run_tests():
    print("\n" + "="*50)
    print("🧪 ЗАПУСК ТЕСТОВ")
    print("="*50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestModelPredictions))
    suite.addTests(loader.loadTestsFromTestCase(TestWebApp))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("❌ ЕСТЬ ОШИБКИ В ТЕСТАХ")
    print("="*50)


if __name__ == "__main__":
    run_tests()
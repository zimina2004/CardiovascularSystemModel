# Создаем базовую модель без сложных вычислений
import pickle

# Простая модель предсказания
class SimpleCardiovascularModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def train(self, data):
        # Подготовка данных
        features = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
                   'cholesterol', 'bmi', 'glucose', 'stress_level', 
                   'exercise_hours', 'smoking', 'family_history', 'gender']
        
        X = data[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение для каждого заболевания
        diseases = ['hypertension', 'coronary_heart_disease', 'diabetes']
        
        for disease in diseases:
            y = data[disease]
            
            # Простая модель логистической регрессии
            model = LogisticRegression(random_state=42, max_iter=500)
            model.fit(X_scaled, y)
            
            self.models[disease] = model
            print(f"Обучена модель для: {disease}")
    
    def predict(self, patient_data):
        # Предсказание для пациента
        X = self.scaler.transform([patient_data])
        
        results = {}
        for disease, model in self.models.items():
            prob = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]
            results[disease] = {'probability': prob, 'prediction': pred}
        
        return results

# Обучение модели
print("Начинаем обучение простой модели...")
model = SimpleCardiovascularModel()

# Используем только первые 500 записей для быстрого обучения
sample_data = cardiovascular_data.head(500)
model.train(sample_data)

print("Модель обучена успешно!")

# Тестирование на примере пациента
test_patient = [45, 140, 90, 75, 220, 26, 110, 5, 2, 1, 0, 1]  # возраст, давление и т.д.
predictions = model.predict(test_patient)

print("\nПрогноз для тестового пациента:")
for disease, result in predictions.items():
    print(f"{disease}: Риск {result['probability']:.2%}, Прогноз: {'Есть риск' if result['prediction'] == 1 else 'Низкий риск'}")

print("\nБазовая модель готова к использованию!")
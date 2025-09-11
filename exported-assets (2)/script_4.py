# Упрощенная и более быстрая версия модели
class CardiovascularSystemModel:
    def __init__(self):
        """Инициализация модели сердечно-сосудистой системы"""
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
            'cholesterol', 'bmi', 'glucose', 'stress_level', 
            'exercise_hours', 'smoking', 'family_history', 'gender'
        ]
        
    def train_disease_models(self, data):
        """Обучение моделей для предсказания заболеваний"""
        diseases = ['hypertension', 'coronary_heart_disease', 'diabetes']
        X = data[self.feature_names]
        
        # Стандартизация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        results = {}
        
        for disease in diseases:
            print(f"Обучение модели для: {disease}")
            y = data[disease]
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Используем Random Forest как лучшую модель
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Оценка
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models[disease] = model
            results[disease] = {
                'accuracy': accuracy,
                'model_type': 'Random Forest'
            }
            
            print(f"Точность для {disease}: {accuracy:.4f}")
        
        return results
    
    def predict_disease_risk(self, patient_data):
        """Предсказание риска заболеваний"""
        processed_data = pd.DataFrame([patient_data])[self.feature_names]
        scaled_data = self.scalers['main'].transform(processed_data)
        
        predictions = {}
        probabilities = {}
        
        for disease, model in self.models.items():
            pred = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0][1]
            
            predictions[disease] = pred
            probabilities[disease] = prob
        
        return predictions, probabilities
    
    def calculate_hemodynamics(self, heart_rate, systolic_bp, diastolic_bp, age, gender):
        """Расчет гемодинамических параметров"""
        # Среднее артериальное давление
        map_value = diastolic_bp + (systolic_bp - diastolic_bp) / 3
        
        # Пульсовое давление
        pulse_pressure = systolic_bp - diastolic_bp
        
        # Ударный объем (упрощенная формула)
        stroke_volume = 70 - (age - 50) * 0.3 + (10 if gender == 1 else 0)
        stroke_volume = max(stroke_volume, 40)  # минимум 40 мл
        
        # Сердечный выброс
        cardiac_output = stroke_volume * heart_rate / 1000  # л/мин
        
        # Общее периферическое сопротивление
        tpr = (map_value * 80) / cardiac_output if cardiac_output > 0 else 0
        
        return {
            'mean_arterial_pressure': map_value,
            'pulse_pressure': pulse_pressure,
            'cardiac_output': cardiac_output,
            'total_peripheral_resistance': tpr,
            'stroke_volume': stroke_volume
        }

# Создание и обучение модели
print("Создание модели сердечно-сосудистой системы...")
cvs_model = CardiovascularSystemModel()
training_results = cvs_model.train_disease_models(cardiovascular_data)

print("\nРезультаты обучения:")
for disease, result in training_results.items():
    print(f"{disease}: {result['accuracy']:.4f}")

print("\nМодель успешно создана и обучена!")
# Класс для моделирования сердечно-сосудистой системы
class CardiovascularSystemModel:
    def __init__(self):
        """
        Инициализация модели сердечно-сосудистой системы
        """
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
            'cholesterol', 'bmi', 'glucose', 'stress_level', 
            'exercise_hours', 'smoking', 'family_history', 'gender'
        ]
        
    def preprocess_data(self, data):
        """
        Предобработка данных
        """
        # Создание копии данных
        processed_data = data[self.feature_names].copy()
        
        # Обработка выбросов с использованием IQR
        for column in processed_data.select_dtypes(include=[np.number]).columns:
            Q1 = processed_data[column].quantile(0.25)
            Q3 = processed_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            processed_data[column] = np.clip(processed_data[column], lower_bound, upper_bound)
        
        return processed_data
    
    def train_disease_models(self, data):
        """
        Обучение моделей для предсказания различных заболеваний
        """
        diseases = ['hypertension', 'coronary_heart_disease', 'diabetes', 'cardiovascular_risk']
        
        # Предобработка данных
        X = self.preprocess_data(data)
        
        # Стандартизация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        results = {}
        
        for disease in diseases:
            print(f"\nОбучение модели для: {disease}")
            y = data[disease]
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Определение моделей для сравнения
            models_to_test = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(),
                'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
            }
            
            best_model = None
            best_score = 0
            model_results = {}
            
            for model_name, model in models_to_test.items():
                # Кросс-валидация
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Обучение модели
                model.fit(X_train, y_train)
                
                # Предсказания
                y_pred = model.predict(X_test)
                
                # Метрики
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                model_results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Выбор лучшей модели
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model = model
            
            # Сохранение лучшей модели
            self.models[disease] = best_model
            results[disease] = model_results
            
            print(f"Лучшая модель для {disease}: {type(best_model).__name__}")
            print(f"Средний CV score: {best_score:.4f}")
        
        return results
    
    def predict_disease_risk(self, patient_data):
        """
        Предсказание риска заболеваний для пациента
        """
        # Предобработка входных данных
        processed_data = self.preprocess_data(pd.DataFrame([patient_data]))
        scaled_data = self.scalers['main'].transform(processed_data)
        
        predictions = {}
        probabilities = {}
        
        for disease, model in self.models.items():
            pred = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0]
            
            predictions[disease] = pred
            probabilities[disease] = prob[1]  # Вероятность наличия заболевания
        
        return predictions, probabilities
    
    def calculate_pressure_dynamics(self, heart_rate, systolic_bp, diastolic_bp, age, gender):
        """
        Расчет динамики давления в сосудах на основе физиологических параметров
        """
        # Расчет среднего артериального давления
        mean_arterial_pressure = diastolic_bp + (systolic_bp - diastolic_bp) / 3
        
        # Расчет пульсового давления
        pulse_pressure = systolic_bp - diastolic_bp
        
        # Расчет сердечного выброса (упрощенная формула)
        stroke_volume = 70 + (age - 50) * (-0.3) + (10 if gender == 1 else 0)  # мл
        cardiac_output = stroke_volume * heart_rate / 1000  # л/мин
        
        # Расчет общего периферического сопротивления
        total_peripheral_resistance = (mean_arterial_pressure * 80) / cardiac_output
        
        # Расчет работы сердца
        cardiac_work = cardiac_output * mean_arterial_pressure * 0.0136  # Вт
        
        return {
            'mean_arterial_pressure': mean_arterial_pressure,
            'pulse_pressure': pulse_pressure,
            'cardiac_output': cardiac_output,
            'total_peripheral_resistance': total_peripheral_resistance,
            'cardiac_work': cardiac_work,
            'stroke_volume': stroke_volume
        }
    
    def simulate_blood_flow(self, vessel_radius, vessel_length, viscosity=0.004, 
                           pressure_gradient=10):
        """
        Симуляция кровотока в сосуде на основе уравнения Пуазейля
        """
        # Закон Пуазейля для ламинарного течения
        flow_rate = (np.pi * vessel_radius**4 * pressure_gradient) / (8 * viscosity * vessel_length)
        
        # Скорость течения
        velocity = flow_rate / (np.pi * vessel_radius**2)
        
        # Напряжение сдвига на стенке сосуда
        wall_shear_stress = (4 * viscosity * velocity) / vessel_radius
        
        return {
            'flow_rate': flow_rate,
            'velocity': velocity,
            'wall_shear_stress': wall_shear_stress,
            'reynolds_number': (2 * vessel_radius * velocity * 1060) / viscosity  # плотность крови ~1060 кг/м³
        }

# Создание и обучение модели
print("Создание модели сердечно-сосудистой системы...")
cvs_model = CardiovascularSystemModel()

# Обучение моделей
training_results = cvs_model.train_disease_models(cardiovascular_data)

print("\nМодель успешно обучена!")
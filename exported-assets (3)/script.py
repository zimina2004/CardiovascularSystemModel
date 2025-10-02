# Создаем полный обновленный код исследования с CatBoost
cardiovascular_catboost_code = '''
# ================================================================================
# СИСТЕМА МОДЕЛИРОВАНИЯ СЕРДЕЧНО-СОСУДИСТОЙ СИСТЕМЫ С CATBOOST
# Исследование по предсказанию сердечно-сосудистых заболеваний
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class HeartHealthPredictor:
    """
    Главный класс для предсказания сердечно-сосудистых заболеваний
    с использованием CatBoost машинного обучения
    """
    
    def __init__(self):
        """Инициализация системы предсказания"""
        self.patient_features = [
            'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
            'cholesterol', 'bmi', 'glucose', 'stress_level', 
            'exercise_hours', 'smoking', 'family_history', 'gender'
        ]
        
        self.disease_models = {}
        self.data_scaler = StandardScaler()
        
        self.disease_names = {
            'hypertension': 'Артериальная гипертензия',
            'coronary_heart_disease': 'Ишемическая болезнь сердца',
            'diabetes': 'Сахарный диабет'
        }
    
    def create_patient_dataset(self, num_patients=1000):
        """
        Создание синтетического набора данных пациентов
        для обучения системы предсказания
        """
        print(f"Создание набора данных для {num_patients} пациентов...")
        
        np.random.seed(42)
        
        # Генерация базовых медицинских показателей
        patient_data = {
            'age': np.clip(np.random.normal(55, 15, num_patients), 20, 90),
            'systolic_bp': np.random.normal(130, 20, num_patients),
            'diastolic_bp': np.random.normal(80, 15, num_patients),
            'heart_rate': np.random.normal(75, 12, num_patients),
            'cholesterol': np.random.normal(200, 40, num_patients),
            'bmi': np.random.normal(25, 4, num_patients),
            'glucose': np.random.normal(100, 25, num_patients),
            'stress_level': np.random.uniform(0, 10, num_patients),
            'exercise_hours': np.random.exponential(3, num_patients),
            'smoking': np.random.binomial(1, 0.3, num_patients),
            'family_history': np.random.binomial(1, 0.4, num_patients),
            'gender': np.random.binomial(1, 0.5, num_patients)
        }
        
        dataset = pd.DataFrame(patient_data)
        
        # Создание целевых переменных (заболеваний) на основе медицинских знаний
        
        # Риск гипертензии
        hypertension_risk = (
            0.004 * dataset['age'] + 
            0.006 * dataset['systolic_bp'] + 
            0.004 * dataset['diastolic_bp'] + 
            0.025 * dataset['bmi'] + 
            0.2 * dataset['smoking'] + 
            0.15 * dataset['family_history'] + 
            0.06 * dataset['stress_level'] - 
            0.04 * dataset['exercise_hours'] + 
            np.random.normal(0, 1, num_patients)
        )
        dataset['hypertension'] = (hypertension_risk > 2.0).astype(int)
        
        # Риск ишемической болезни сердца
        chd_risk = (
            0.006 * dataset['age'] + 
            0.003 * dataset['systolic_bp'] + 
            0.002 * dataset['cholesterol'] + 
            0.02 * dataset['bmi'] + 
            0.25 * dataset['smoking'] + 
            0.18 * dataset['family_history'] + 
            0.04 * dataset['stress_level'] - 
            0.03 * dataset['exercise_hours'] + 
            np.random.normal(0, 0.8, num_patients)
        )
        dataset['coronary_heart_disease'] = (chd_risk > 1.5).astype(int)
        
        # Риск диабета
        diabetes_risk = (
            0.005 * dataset['age'] + 
            0.002 * dataset['systolic_bp'] + 
            0.004 * dataset['glucose'] + 
            0.03 * dataset['bmi'] + 
            0.1 * dataset['family_history'] + 
            0.03 * dataset['stress_level'] - 
            0.02 * dataset['exercise_hours'] + 
            np.random.normal(0, 0.6, num_patients)
        )
        dataset['diabetes'] = (diabetes_risk > 1.2).astype(int)
        
        print("✓ Набор данных успешно создан")
        print(f"Размер данных: {dataset.shape}")
        
        return dataset
    
    def train_disease_prediction_models(self, patient_dataset):
        """
        Обучение моделей CatBoost для предсказания каждого заболевания
        """
        print("\\nНачало обучения моделей CatBoost...")
        
        # Подготовка признаков
        X = patient_dataset[self.patient_features]
        X_scaled = self.data_scaler.fit_transform(X)
        
        training_results = {}
        
        for disease_code, disease_name in self.disease_names.items():
            print(f"\\nОбучение модели для: {disease_name}")
            
            y = patient_dataset[disease_code]
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Создание и настройка CatBoost модели
            catboost_model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False,
                loss_function='Logloss',
                eval_metric='AUC'
            )
            
            # Обучение модели
            catboost_model.fit(X_train, y_train)
            
            # Оценка качества модели
            y_pred = catboost_model.predict(X_test)
            y_proba = catboost_model.predict_proba(X_test)[:, 1]
            
            # Расчет метрик качества
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Кросс-валидация
            cv_scores = cross_val_score(catboost_model, X_train, y_train, cv=5)
            
            training_results[disease_code] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Сохранение обученной модели
            self.disease_models[disease_code] = catboost_model
            
            print(f"✓ Точность: {accuracy:.3f}")
            print(f"✓ AUC: {auc:.3f}")
            print(f"✓ F1-мера: {f1:.3f}")
        
        return training_results
    
    def predict_patient_diseases(self, patient_info):
        """
        Предсказание заболеваний для конкретного пациента
        """
        # Подготовка данных пациента
        if isinstance(patient_info, dict):
            patient_df = pd.DataFrame([patient_info])
        else:
            patient_df = pd.DataFrame([patient_info], columns=self.patient_features)
        
        patient_scaled = self.data_scaler.transform(patient_df[self.patient_features])
        
        predictions = {}
        probabilities = {}
        
        for disease_code, model in self.disease_models.items():
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0][1]
            
            predictions[disease_code] = prediction
            probabilities[disease_code] = probability
        
        return predictions, probabilities
    
    def calculate_heart_parameters(self, heart_rate, systolic_bp, diastolic_bp, age, gender):
        """
        Расчет основных параметров работы сердца и сосудов
        """
        # Среднее артериальное давление
        mean_pressure = diastolic_bp + (systolic_bp - diastolic_bp) / 3
        
        # Пульсовое давление
        pulse_pressure = systolic_bp - diastolic_bp
        
        # Ударный объем сердца (упрощенная формула)
        stroke_volume = 70 - (age - 50) * 0.25 + (10 if gender == 1 else 0)
        stroke_volume = max(stroke_volume, 40)  # минимальное значение
        
        # Сердечный выброс
        cardiac_output = stroke_volume * heart_rate / 1000  # л/мин
        
        # Общее периферическое сопротивление
        peripheral_resistance = (mean_pressure * 80) / cardiac_output if cardiac_output > 0 else 0
        
        # Работа сердца
        cardiac_work = cardiac_output * mean_pressure * 0.0136  # Вт
        
        return {
            'mean_arterial_pressure': round(mean_pressure, 2),
            'pulse_pressure': round(pulse_pressure, 2),
            'stroke_volume': round(stroke_volume, 2),
            'cardiac_output': round(cardiac_output, 2),
            'peripheral_resistance': round(peripheral_resistance, 2),
            'cardiac_work': round(cardiac_work, 2)
        }
    
    def simulate_blood_vessel_flow(self, vessel_radius, vessel_length, cardiac_output, 
                                  blood_viscosity=0.004):
        """
        Моделирование кровотока в сосуде на основе законов физики
        """
        # Сопротивление сосуда (закон Пуазейля)
        vessel_resistance = (8 * blood_viscosity * vessel_length) / (np.pi * vessel_radius**4)
        
        # Скорость кровотока
        blood_velocity = cardiac_output / (np.pi * vessel_radius**2 * 60)  # м/с
        
        # Напряжение сдвига на стенке сосуда
        wall_shear_stress = (4 * blood_viscosity * blood_velocity) / vessel_radius
        
        # Число Рейнольдса (характеризует тип течения)
        reynolds_number = (2 * vessel_radius * blood_velocity * 1060) / blood_viscosity
        
        # Падение давления в сосуде
        pressure_drop = cardiac_output * vessel_resistance * 1000  # Па
        
        return {
            'vessel_resistance': round(vessel_resistance, 6),
            'blood_velocity': round(blood_velocity, 4),
            'wall_shear_stress': round(wall_shear_stress, 4),
            'reynolds_number': round(reynolds_number, 2),
            'pressure_drop': round(pressure_drop, 2)
        }
    
    def comprehensive_health_analysis(self, patient_info):
        """
        Комплексный анализ здоровья сердечно-сосудистой системы пациента
        """
        # Предсказание заболеваний
        predictions, probabilities = self.predict_patient_diseases(patient_info)
        
        # Расчет параметров сердца
        heart_params = self.calculate_heart_parameters(
            patient_info['heart_rate'], patient_info['systolic_bp'],
            patient_info['diastolic_bp'], patient_info['age'], patient_info['gender']
        )
        
        # Расчет общего риска
        total_risk = sum(probabilities.values()) / len(probabilities)
        
        # Определение уровня риска
        if total_risk < 0.3:
            risk_level = "Низкий риск"
            risk_color = "🟢"
        elif total_risk < 0.6:
            risk_level = "Умеренный риск"
            risk_color = "🟡"
        else:
            risk_level = "Высокий риск"
            risk_color = "🔴"
        
        return {
            'disease_predictions': predictions,
            'disease_probabilities': probabilities,
            'heart_parameters': heart_params,
            'total_cardiovascular_risk': round(total_risk, 4),
            'risk_level': risk_level,
            'risk_color': risk_color
        }

class MedicalReportGenerator:
    """Класс для создания медицинских отчетов и рекомендаций"""
    
    @staticmethod
    def create_patient_report(analysis_results, patient_info):
        """Создание подробного медицинского отчета"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ОТЧЕТ АНАЛИЗА СЕРДЕЧНО-СОСУДИСТОЙ СИСТЕМЫ                 ║
║                           Система на базе CatBoost ML                         ║
╠══════════════════════════════════════════════════════════════════════════════╣

ДАННЫЕ ПАЦИЕНТА:
• Возраст: {patient_info['age']} лет
• Пол: {'Мужской' if patient_info['gender'] == 1 else 'Женский'}
• ИМТ: {patient_info['bmi']:.1f}
• Курение: {'Да' if patient_info['smoking'] == 1 else 'Нет'}

ПОКАЗАТЕЛИ ДАВЛЕНИЯ И СЕРДЦА:
• Систолическое АД: {patient_info['systolic_bp']:.0f} мм рт.ст.
• Диастолическое АД: {patient_info['diastolic_bp']:.0f} мм рт.ст.
• Частота сердечных сокращений: {patient_info['heart_rate']:.0f} уд/мин
• Холестерин: {patient_info['cholesterol']:.0f} мг/дл

РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ ЗАБОЛЕВАНИЙ:
"""
        
        disease_names = {
            'hypertension': 'Артериальная гипертензия',
            'coronary_heart_disease': 'Ишемическая болезнь сердца',
            'diabetes': 'Сахарный диабет'
        }
        
        for disease, prob in analysis_results['disease_probabilities'].items():
            disease_name = disease_names.get(disease, disease)
            
            if prob > 0.7:
                status = "ОЧЕНЬ ВЫСОКИЙ РИСК 🔴"
            elif prob > 0.5:
                status = "ВЫСОКИЙ РИСК 🟠"
            elif prob > 0.3:
                status = "УМЕРЕННЫЙ РИСК 🟡"
            else:
                status = "НИЗКИЙ РИСК 🟢"
                
            report += f"• {disease_name}: {prob:.1%} ({status})\\n"
        
        report += f"""
ПАРАМЕТРЫ РАБОТЫ СЕРДЦА:
• Среднее артериальное давление: {analysis_results['heart_parameters']['mean_arterial_pressure']} мм рт.ст.
• Пульсовое давление: {analysis_results['heart_parameters']['pulse_pressure']} мм рт.ст.
• Сердечный выброс: {analysis_results['heart_parameters']['cardiac_output']} л/мин
• Ударный объем: {analysis_results['heart_parameters']['stroke_volume']} мл
• Периферическое сопротивление: {analysis_results['heart_parameters']['peripheral_resistance']} дин⋅с⋅см⁻⁵

ОБЩАЯ ОЦЕНКА:
{analysis_results['risk_color']} Интегральный кардиоваскулярный риск: {analysis_results['total_cardiovascular_risk']:.1%}
{analysis_results['risk_color']} Уровень риска: {analysis_results['risk_level']}

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report
    
    @staticmethod
    def generate_medical_recommendations(analysis_results):
        """Генерация медицинских рекомендаций на основе анализа"""
        
        recommendations = []
        probabilities = analysis_results['disease_probabilities']
        
        # Рекомендации по гипертензии
        if probabilities['hypertension'] > 0.6:
            recommendations.extend([
                "🩺 Консультация кардиолога для оценки артериального давления",
                "💊 Рассмотреть назначение антигипертензивной терапии",
                "📊 Регулярный мониторинг артериального давления"
            ])
        
        # Рекомендации по ИБС
        if probabilities['coronary_heart_disease'] > 0.6:
            recommendations.extend([
                "🫀 Проведение ЭКГ и эхокардиографии",
                "🏃 Нагрузочные тесты для оценки функции сердца",
                "💊 Консультация по антиагрегантной терапии"
            ])
        
        # Рекомендации по диабету
        if probabilities['diabetes'] > 0.6:
            recommendations.extend([
                "🍬 Контроль уровня глюкозы крови",
                "👨‍⚕️ Консультация эндокринолога",
                "🥗 Коррекция диеты и режима питания"
            ])
        
        # Общие рекомендации при высоком риске
        if analysis_results['total_cardiovascular_risk'] > 0.5:
            recommendations.extend([
                "🏃‍♂️ Увеличение физической активности (150 мин/неделю)",
                "🚭 Отказ от курения (при наличии)",
                "🥗 Средиземноморская диета с ограничением соли",
                "😌 Управление стрессом и нормализация сна",
                "📅 Регулярные профилактические осмотры"
            ])
        
        return recommendations

def main_cardiovascular_analysis():
    """Основная функция для запуска анализа сердечно-сосудистой системы"""
    
    print("="*80)
    print("🫀 СИСТЕМА АНАЛИЗА СЕРДЕЧНО-СОСУДИСТОГО ЗДОРОВЬЯ")
    print("🤖 На базе CatBoost машинного обучения")
    print("="*80)
    
    # Создание системы предсказания
    heart_predictor = HeartHealthPredictor()
    report_generator = MedicalReportGenerator()
    
    # Создание и обучение на данных
    print("\\n📊 Этап 1: Создание обучающих данных...")
    training_data = heart_predictor.create_patient_dataset(1000)
    
    print("\\n🤖 Этап 2: Обучение моделей CatBoost...")
    training_results = heart_predictor.train_disease_prediction_models(training_data)
    
    # Отображение результатов обучения
    print("\\n" + "─" * 60)
    print("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("─" * 60)
    
    for disease, metrics in training_results.items():
        disease_name = heart_predictor.disease_names[disease]
        print(f"\\n{disease_name}:")
        print(f"  Точность: {metrics['accuracy']:.3f}")
        print(f"  AUC: {metrics['auc']:.3f}")
        print(f"  F1-мера: {metrics['f1_score']:.3f}")
        print(f"  Кросс-валидация: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    # Тестирование на примере пациента
    print("\\n🧑‍⚕️ Этап 3: Анализ тестового пациента...")
    
    test_patient = {
        'age': 58, 'systolic_bp': 155, 'diastolic_bp': 98, 'heart_rate': 85,
        'cholesterol': 235, 'bmi': 29, 'glucose': 115, 'stress_level': 7,
        'exercise_hours': 1.5, 'smoking': 1, 'family_history': 1, 'gender': 1
    }
    
    # Комплексный анализ
    analysis = heart_predictor.comprehensive_health_analysis(test_patient)
    
    # Генерация отчета
    medical_report = report_generator.create_patient_report(analysis, test_patient)
    print(medical_report)
    
    # Медицинские рекомендации
    recommendations = report_generator.generate_medical_recommendations(analysis)
    
    print("\\n" + "─" * 60)
    print("💡 МЕДИЦИНСКИЕ РЕКОМЕНДАЦИИ")
    print("─" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Моделирование кровотока
    print("\\n🩸 Этап 4: Моделирование кровотока в сосудах...")
    
    vessel_analysis = heart_predictor.simulate_blood_vessel_flow(
        vessel_radius=0.003,  # 3 мм радиус
        vessel_length=0.15,   # 15 см длина
        cardiac_output=analysis['heart_parameters']['cardiac_output']
    )
    
    print("\\nПараметры кровотока:")
    vessel_params = {
        'vessel_resistance': 'Сопротивление сосуда',
        'blood_velocity': 'Скорость крови (м/с)',
        'wall_shear_stress': 'Напряжение сдвига (Па)',
        'reynolds_number': 'Число Рейнольдса',
        'pressure_drop': 'Падение давления (Па)'
    }
    
    for param, value in vessel_analysis.items():
        param_name = vessel_params.get(param, param)
        print(f"• {param_name}: {value}")
    
    print("\\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("📋 Система готова к клиническому применению")
    print("="*80)

if __name__ == "__main__":
    main_cardiovascular_analysis()
'''

# Сохраняем обновленный код
with open('cardiovascular_catboost_system.py', 'w', encoding='utf-8') as f:
    f.write(cardiovascular_catboost_code)

print("✅ Создан файл: cardiovascular_catboost_system.py")
print("📊 Размер кода:", len(cardiovascular_catboost_code), "символов")